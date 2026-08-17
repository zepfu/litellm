"""CFG-004 endpoint: POST /aawm/alias-routing/cooldowns/clear tests.

Covers: strict schema validation, auth/master key via real hash, topology gate,
input exclusivity, ingress mismatch, anthropic_route_family projection,
alias/exact active-snapshot resolution via lane-identity index (no fabricated
keys), multi-candidate alias retention, exact ambiguity rejection, fail-closed
on missing DualCache/Redis errors, idempotent not_active after authoritative
absence, successful clear, endpoint-owned family lock serialization,
reservation-first sequencing, object-identity safe completion,
postcondition/invalidation failure, unrelated state/affinity preservation,
audit events, error redaction, and no provider traffic.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.candidate_loop import (
    _default_lane_identity_hash,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_apply import (
    execute_cooldown_publication_transaction,
    resolve_lane_identity_hash,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
    CooldownClearRequest,
    _check_admin_auth,
    _check_topology_gate,
    _emit_audit_event,
    _execute_clear,
    _hydrate_identities_from_durable,
    _hydrate_from_local_index,
    _parse_and_validate_request,
    _ResolvedTarget,
    _resolve_target_from_active_snapshot,
    handle_cooldown_clear,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_snapshot import (
    AliasReference,
    RoutingAlias,
    RoutingCandidate,
    RoutingSnapshot,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.interfaces import (
    CooldownPublicationPlan,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    AliasRoutingStateManager,
    ClaimOutcome,
)

# ---------------------------------------------------------------------------
# Module paths for patching
# ---------------------------------------------------------------------------

_CLEAR_MOD = "litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear"
_APPLY_MOD = "litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_apply"
_DURABLE_MOD = "litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(
    *,
    provider: str = "openai",
    model: str = "gpt-4o",
    route_family: str = "codex_openai_responses",
    anthropic_route_family: str | None = None,
    priority: int = 10,
) -> RoutingCandidate:
    return RoutingCandidate(
        provider=provider,
        model=model,
        route_family=route_family,
        priority=priority,
        weight=1.0,
        tui_attached=None,
        schedule=None,
        anthropic_route_family=anthropic_route_family,
    )


def _make_snapshot(
    *,
    alias_name: str = "test-alias",
    provider: str = "openai",
    model: str = "gpt-4o",
    route_family: str = "codex_openai_responses",
    anthropic_route_family: str | None = None,
    extra_candidates: list[RoutingCandidate] | None = None,
) -> RoutingSnapshot:
    candidates = [
        _make_candidate(
            provider=provider,
            model=model,
            route_family=route_family,
            anthropic_route_family=anthropic_route_family,
        ),
    ]
    if extra_candidates:
        candidates.extend(extra_candidates)
    alias = RoutingAlias(
        name=alias_name,
        distribution_strategy=None,
        candidates=tuple(candidates),
    )
    return RoutingSnapshot(
        aliases={alias_name: alias},
        config_epoch=1,
        config_hash="abc123def456",
        config_version="abc123def456",
    )


def _make_request(body: dict) -> MagicMock:
    request = MagicMock()
    request.json = AsyncMock(return_value=body)
    return request


def _make_real_admin_key_dict(*, api_key: str = "sk-master-secret") -> MagicMock:
    """Build a UserAPIKeyAuth-like mock using the REAL hash method."""
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

    mock = MagicMock()
    mock.user_role = LitellmUserRoles.PROXY_ADMIN
    # Use the actual hashing method -- no mock.
    mock.token = UserAPIKeyAuth._safe_hash_litellm_api_key(api_key)
    return mock


def _seed_lane_keys(
    mgr: AliasRoutingStateManager,
    *,
    family: str,
    identity_hash: str,
    lane_keys: list[str],
    cooldown_seconds: float = 300.0,
) -> None:
    """Register lane keys in the index and seed local cooldown state."""
    mgr.lane_identity_index.register_batch(
        identity_hash=identity_hash,
        lane_keys=lane_keys,
    )
    family_state = mgr.family(family)
    for key in lane_keys:
        family_state.cooldown_until_monotonic_by_key[key] = (
            time.monotonic() + cooldown_seconds
        )


def _resolve_identity_hash(
    *,
    provider: str,
    model: str,
    route_family: str,
) -> str:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_apply import (
        resolve_lane_identity_hash,
    )

    return resolve_lane_identity_hash(
        candidate={
            "provider": provider,
            "model": model,
            "route_family": route_family,
        }
    )


def _make_mock_dual_cache() -> MagicMock:
    mock = MagicMock()
    mock.redis_cache = MagicMock()
    return mock


def _make_identity_inspection(
    *,
    exists: bool = True,
    members: frozenset | None = None,
    cardinality: int = 1,
    ttl_remaining_seconds: float | None = 300.0,
) -> MagicMock:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        IdentitySetInspection,
    )

    return IdentitySetInspection(
        identity_key="idkey",
        exists=exists,
        members=members if members is not None else frozenset(),
        cardinality=cardinality,
        ttl_remaining_seconds=ttl_remaining_seconds,
    )


def _make_txn_result(
    *,
    keys_deleted: int = 1,
    members_removed: int = 1,
) -> MagicMock:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        ClearTransactionJournal,
        ClearTransactionResult,
    )

    return ClearTransactionResult(
        transaction_id="txn123",
        phase="CLEAR_COMMITTED",
        journal=ClearTransactionJournal(
            transaction_id="txn123",
            phase="CLEAR_COMMITTED",
            alias_family="codex",
            identity_hash="idhash",
            cooldown_keys=["k"],
            lane_members=["k"],
            expected_members=["k"],
            identity_key="idkey",
            receipt_key="rkey",
            receipt_ttl=300,
        ),
        keys_deleted=keys_deleted,
        members_removed=members_removed,
    )


def _common_patches(
    *,
    snapshot: RoutingSnapshot,
    mgr: AliasRoutingStateManager,
    dual_cache: MagicMock | None = None,
    master_key: str = "sk-master-secret",
):
    """Return a context-manager stack for common handle_cooldown_clear patches."""
    if dual_cache is None:
        dual_cache = _make_mock_dual_cache()
    return (
        patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
        patch("litellm.proxy.proxy_server.master_key", master_key),
        patch(f"{_CLEAR_MOD}.alias_routing_state", mgr),
        patch(
            f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
            return_value=dual_cache,
        ),
    )


@pytest.fixture()
def fresh_manager() -> AliasRoutingStateManager:
    return AliasRoutingStateManager()


@pytest.fixture(autouse=True)
def _topology_gate_open(monkeypatch):
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_COOLDOWN_CLEAR_SINGLE_WORKER", "1")


# ---------------------------------------------------------------------------
# CFG-019 alias-scoped control-plane identity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shared_route_aliases_keep_control_plane_membership_isolated(  # noqa: PLR0915
    fresh_manager: AliasRoutingStateManager,
) -> None:
    shared_candidate = _make_candidate()
    snapshot = RoutingSnapshot(
        aliases={
            alias_name: RoutingAlias(
                name=alias_name,
                distribution_strategy=None,
                candidates=(shared_candidate,),
            )
            for alias_name in ("alias-a", "alias-b")
        },
        config_epoch=1,
        config_hash="shared-config-hash",
        config_version="shared-confi",
    )

    with patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot):
        target_a = _resolve_target_from_active_snapshot(
            CooldownClearRequest(alias="alias-a", ingress="codex")
        )
        target_b = _resolve_target_from_active_snapshot(
            CooldownClearRequest(alias="alias-b", ingress="codex")
        )

    candidate_base = {
        "provider": shared_candidate.provider,
        "model": shared_candidate.model,
        "route_family": shared_candidate.route_family,
    }
    candidate_a = {
        **candidate_base,
        "cooldown_identity_tag": (
            "alias:alias-a:openai:gpt-4o:codex_openai_responses"
        ),
    }
    candidate_b = {
        **candidate_base,
        "cooldown_identity_tag": (
            "alias:alias-b:openai:gpt-4o:codex_openai_responses"
        ),
    }
    identity_a = target_a.identities[0].identity_hash
    identity_b = target_b.identities[0].identity_hash
    assert identity_a == resolve_lane_identity_hash(candidate=candidate_a)
    assert identity_b == resolve_lane_identity_hash(candidate=candidate_b)
    assert identity_a != identity_b
    assert _default_lane_identity_hash(candidate=candidate_a) == identity_a
    assert _default_lane_identity_hash(candidate=candidate_b) == identity_b

    lane_a = "halias-a:openai:gpt-4o:lane"
    lane_b = "halias-b:openai:gpt-4o:lane"
    dual_cache = _make_mock_dual_cache()
    publish_mock = AsyncMock(
        side_effect=[MagicMock(journal=MagicMock()), MagicMock(journal=MagicMock())]
    )

    def publish_memory(*, keys, seconds):
        for key in keys:
            fresh_manager.codex.set_cooldown_memory(key, seconds)

    with (
        patch(f"{_APPLY_MOD}._state_manager", fresh_manager),
        patch(
            f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache",
            return_value=dual_cache,
        ),
        patch(f"{_DURABLE_MOD}.publish_cooldown_transaction", publish_mock),
    ):
        for candidate, lane_key in (
            (candidate_a, lane_a),
            (candidate_b, lane_b),
        ):
            await execute_cooldown_publication_transaction(
                alias_family="codex",
                candidate=candidate,
                plan=CooldownPublicationPlan(
                    memory_keys=(lane_key,),
                    durable_keys=(lane_key,),
                    duration_seconds=300.0,
                    applied_scope="candidate",
                ),
                publish_cooldown_memory_fn=publish_memory,
                persist_cooldown_fn=AsyncMock(),
            )

    assert [
        call.kwargs["identity_hash"] for call in publish_mock.await_args_list
    ] == [identity_a, identity_b]
    assert fresh_manager.lane_identity_index.lanes_for(identity_a) == frozenset(
        {lane_a}
    )
    assert fresh_manager.lane_identity_index.lanes_for(identity_b) == frozenset(
        {lane_b}
    )

    registry = fresh_manager.publication_intents
    reservation = registry.create_clear_reservation(
        alias_family="codex",
        identity_hashes=frozenset({identity_a}),
        cooldown_keys=frozenset({lane_a}),
    )
    blocked = registry.claim_publication_or_wait(
        alias_family="codex",
        cooldown_keys=frozenset({lane_a}),
        identity_hash=identity_a,
    )
    isolated = registry.claim_publication_or_wait(
        alias_family="codex",
        cooldown_keys=frozenset({lane_b}),
        identity_hash=identity_b,
    )
    assert blocked.outcome is ClaimOutcome.BLOCKED_BY_CLEAR
    assert isolated.outcome is ClaimOutcome.LEADER
    assert isolated.intent is not None
    registry.release_claim(isolated.intent)

    _hydrate_from_local_index(target_a, fresh_manager)
    _hydrate_from_local_index(target_b, fresh_manager)
    inspections = {
        identity_a: _make_identity_inspection(
            members=frozenset({lane_a}), cardinality=1
        ),
        identity_b: _make_identity_inspection(
            members=frozenset({lane_b}), cardinality=1
        ),
    }

    async def inspect_identity(*, identity_hash, **_kwargs):
        return inspections[identity_hash]

    with (
        patch(
            f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
            return_value=dual_cache,
        ),
        patch(
            f"{_CLEAR_MOD}.inspect_identity_set",
            new_callable=AsyncMock,
            side_effect=inspect_identity,
        ) as inspect_mock,
    ):
        await _hydrate_identities_from_durable(target_a)
        await _hydrate_identities_from_durable(target_b)

    assert {
        call.kwargs["identity_hash"] for call in inspect_mock.await_args_list
    } == {identity_a, identity_b}
    assert target_a.identities[0].lane_keys == [lane_a]
    assert target_b.identities[0].lane_keys == [lane_b]

    durable_clear = AsyncMock(
        return_value=MagicMock(keys_deleted=1, members_removed=1)
    )
    with (
        patch(f"{_CLEAR_MOD}._execute_durable_clear", durable_clear),
        patch(
            f"{_CLEAR_MOD}._verify_postconditions",
            new_callable=AsyncMock,
        ),
    ):
        assert await _execute_clear(target_a, fresh_manager) == (1, 1)

    durable_clear.assert_awaited_once_with(
        family="codex",
        identity_hash=identity_a,
        cooldown_keys=[lane_a],
        lane_members=[lane_a],
    )
    assert fresh_manager.lane_identity_index.lanes_for(identity_a) == frozenset()
    assert fresh_manager.lane_identity_index.lanes_for(identity_b) == frozenset(
        {lane_b}
    )
    assert fresh_manager.codex.get_memory_cooldown_remaining(lane_a) == 0.0
    assert fresh_manager.codex.get_memory_cooldown_remaining(lane_b) > 0.0
    registry.complete_clear_reservation(reservation)


# ---------------------------------------------------------------------------
# Strict schema validation tests (Acceptance #5)
# ---------------------------------------------------------------------------


class TestStrictSchemaValidation:
    def test_alias_only_accepted(self):
        req = _parse_and_validate_request({"alias": "test", "ingress": "codex"})
        assert req.alias == "test"
        assert req.provider is None

    def test_exact_only_accepted(self):
        req = _parse_and_validate_request(
            {"provider": "openai", "model": "gpt-4o", "ingress": "codex"}
        )
        assert req.provider == "openai"
        assert req.model == "gpt-4o"
        assert req.alias is None

    def test_both_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_request(
                {"alias": "test", "provider": "openai", "model": "gpt-4o", "ingress": "codex"}
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "ambiguous_target"

    def test_neither_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_request({"ingress": "codex"})
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "missing_target"

    def test_provider_without_model_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_request({"provider": "openai", "ingress": "codex"})
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "incomplete_target"

    def test_model_without_provider_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_request({"model": "gpt-4o", "ingress": "codex"})
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "incomplete_target"

    def test_extra_fields_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_request(
                {"alias": "test", "ingress": "codex", "extra_field": "bad"}
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "unexpected_fields"

    def test_raw_key_field_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_request(
                {"alias": "test", "ingress": "codex", "key": "raw-value"}
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "unexpected_fields"

    def test_hash_field_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_request(
                {"alias": "test", "ingress": "codex", "hash": "abc"}
            )
        assert exc_info.value.status_code == 400

    def test_namespace_field_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_request(
                {"alias": "test", "ingress": "codex", "namespace": "ns"}
            )
        assert exc_info.value.status_code == 400

    def test_pattern_field_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_request(
                {"alias": "test", "ingress": "codex", "pattern": "*"}
            )
        assert exc_info.value.status_code == 400

    def test_global_field_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_request(
                {"alias": "test", "ingress": "codex", "global": True}
            )
        assert exc_info.value.status_code == 400

    def test_non_string_alias_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_request({"alias": 123, "ingress": "codex"})
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "invalid_field_type"

    def test_non_string_ingress_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_request({"alias": "test", "ingress": ["codex"]})
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "invalid_field_type"

    def test_invalid_ingress_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_request({"alias": "test", "ingress": "openai"})
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "invalid_ingress"

    def test_missing_ingress_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_request({"alias": "test"})
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "invalid_ingress"

    def test_codex_ingress_accepted(self):
        req = _parse_and_validate_request({"alias": "test", "ingress": "codex"})
        assert req.ingress == "codex"

    def test_anthropic_ingress_accepted(self):
        req = _parse_and_validate_request({"alias": "test", "ingress": "anthropic"})
        assert req.ingress == "anthropic"


# ---------------------------------------------------------------------------
# Topology gate tests
# ---------------------------------------------------------------------------


class TestTopologyGate:
    def test_gate_open_with_1(self, monkeypatch):
        monkeypatch.setenv("AAWM_ALIAS_ROUTING_COOLDOWN_CLEAR_SINGLE_WORKER", "1")
        _check_topology_gate()

    def test_gate_closed_absent(self, monkeypatch):
        monkeypatch.delenv("AAWM_ALIAS_ROUTING_COOLDOWN_CLEAR_SINGLE_WORKER", raising=False)
        with pytest.raises(HTTPException) as exc_info:
            _check_topology_gate()
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail["error"] == "topology_gate_closed"

    def test_gate_closed_false(self, monkeypatch):
        monkeypatch.setenv("AAWM_ALIAS_ROUTING_COOLDOWN_CLEAR_SINGLE_WORKER", "false")
        with pytest.raises(HTTPException) as exc_info:
            _check_topology_gate()
        assert exc_info.value.status_code == 503

    def test_gate_closed_zero(self, monkeypatch):
        monkeypatch.setenv("AAWM_ALIAS_ROUTING_COOLDOWN_CLEAR_SINGLE_WORKER", "0")
        with pytest.raises(HTTPException) as exc_info:
            _check_topology_gate()
        assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# Auth tests (Acceptance #8: real hash, no mock of hash method)
# ---------------------------------------------------------------------------


class TestAuth:
    def test_non_admin_rejected(self):
        from litellm.proxy._types import LitellmUserRoles

        mock = MagicMock()
        mock.user_role = LitellmUserRoles.INTERNAL_USER
        with pytest.raises(HTTPException) as exc_info:
            _check_admin_auth(mock)
        assert exc_info.value.status_code == 403
        assert exc_info.value.detail["error"] == "forbidden"

    def test_admin_with_real_master_key_accepted(self):
        """Real master key hash succeeds without mocking _safe_hash_litellm_api_key."""
        admin = _make_real_admin_key_dict(api_key="sk-master-secret")
        with patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"):
            _check_admin_auth(admin)  # Should not raise

    def test_delegated_admin_key_rejected(self):
        """PROXY_ADMIN virtual/admin key with different secret fails closed."""
        admin = _make_real_admin_key_dict(api_key="sk-delegated-admin-key")
        with patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"):
            with pytest.raises(HTTPException) as exc_info:
                _check_admin_auth(admin)
            assert exc_info.value.status_code == 403
            assert exc_info.value.detail["error"] == "forbidden"

    def test_no_master_key_configured_fails_closed(self):
        admin = _make_real_admin_key_dict()
        with patch("litellm.proxy.proxy_server.master_key", None):
            with pytest.raises(HTTPException) as exc_info:
                _check_admin_auth(admin)
            assert exc_info.value.status_code == 503
            assert exc_info.value.detail["error"] == "auth_unavailable"

    def test_absent_token_fails_closed(self):
        from litellm.proxy._types import LitellmUserRoles

        mock = MagicMock()
        mock.user_role = LitellmUserRoles.PROXY_ADMIN
        mock.token = None
        with patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"):
            with pytest.raises(HTTPException) as exc_info:
                _check_admin_auth(mock)
            assert exc_info.value.status_code == 403

    def test_malformed_token_fails_closed(self):
        from litellm.proxy._types import LitellmUserRoles

        mock = MagicMock()
        mock.user_role = LitellmUserRoles.PROXY_ADMIN
        mock.token = "not-a-real-hash"
        with patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"):
            with pytest.raises(HTTPException) as exc_info:
                _check_admin_auth(mock)
            assert exc_info.value.status_code == 403

    def test_error_never_exposes_secrets(self):
        from litellm.proxy._types import LitellmUserRoles

        mock = MagicMock()
        mock.user_role = LitellmUserRoles.INTERNAL_USER
        with pytest.raises(HTTPException) as exc_info:
            _check_admin_auth(mock)
        detail_str = str(exc_info.value.detail)
        assert "sk-" not in detail_str
        assert "Bearer" not in detail_str
        assert "authorization" not in detail_str.lower()


# ---------------------------------------------------------------------------
# Snapshot resolution tests (Acceptance #1, #2)
# ---------------------------------------------------------------------------


class TestSnapshotResolution:
    def test_no_active_snapshot_fails_closed(self, fresh_manager):
        req = CooldownClearRequest(alias="test", ingress="codex")
        with patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=None):
            with pytest.raises(HTTPException) as exc_info:
                _resolve_target_from_active_snapshot(req)
            assert exc_info.value.status_code == 503
            assert exc_info.value.detail["error"] == "no_active_snapshot"

    def test_alias_not_found_returns_404(self, fresh_manager):
        snapshot = _make_snapshot(alias_name="other")
        req = CooldownClearRequest(alias="nonexistent", ingress="codex")
        with patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot):
            with pytest.raises(HTTPException) as exc_info:
                _resolve_target_from_active_snapshot(req)
            assert exc_info.value.status_code == 404
            assert exc_info.value.detail["error"] == "alias_not_found"

    def test_exact_not_found_returns_404(self, fresh_manager):
        snapshot = _make_snapshot(provider="openai", model="gpt-4o")
        req = CooldownClearRequest(provider="anthropic", model="claude-3", ingress="codex")
        with patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot):
            with pytest.raises(HTTPException) as exc_info:
                _resolve_target_from_active_snapshot(req)
            assert exc_info.value.status_code == 404
            assert exc_info.value.detail["error"] == "target_not_found"

    def test_pure_snapshot_resolution_returns_no_lane_keys(self, fresh_manager):
        """Pure snapshot resolution derives identity_hash but NO lane keys.
        Lane keys are populated later by _hydrate_from_local_index (Fix 1)."""
        snapshot = _make_snapshot()
        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        real_key = "h{epoch}:openai:gpt-4o:chatgpt-account:acct1".format(
            epoch=snapshot.config_hash
        )
        _seed_lane_keys(
            fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[real_key]
        )
        req = CooldownClearRequest(alias="test-alias", ingress="codex")
        with patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot):
            target = _resolve_target_from_active_snapshot(req)
            assert target.family == "codex"
            assert target.target_description == "alias:test-alias"
            assert len(target.identities) == 1
            assert target.identities[0].identity_hash == id_hash
            # Pure snapshot phase: NO lane keys yet.
            assert target.identities[0].lane_keys == []
            assert target.all_cooldown_keys == []

    def test_hydrate_from_local_index_populates_lane_keys(self, fresh_manager):
        """_hydrate_from_local_index populates lane_keys from the index
        AFTER reservation would be established (Fix 1)."""
        snapshot = _make_snapshot()
        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        real_key = "h{epoch}:openai:gpt-4o:chatgpt-account:acct1".format(
            epoch=snapshot.config_hash
        )
        _seed_lane_keys(
            fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[real_key]
        )
        req = CooldownClearRequest(alias="test-alias", ingress="codex")
        with patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot):
            target = _resolve_target_from_active_snapshot(req)
            # Before hydration: empty.
            assert target.identities[0].lane_keys == []
            # After hydration: populated from index.
            _hydrate_from_local_index(target, fresh_manager)
            assert target.identities[0].lane_keys == [real_key]
            # No fabricated __default__ keys.
            for key in target.all_cooldown_keys:
                assert "__default__" not in key

    def test_alias_retains_all_candidates(self, fresh_manager):
        """Multi-candidate alias retains all active candidates."""
        cand2 = _make_candidate(
            provider="xai", model="grok-3", route_family="codex_xai_responses"
        )
        snapshot = _make_snapshot(extra_candidates=[cand2])
        req = CooldownClearRequest(alias="test-alias", ingress="codex")
        with patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot):
            target = _resolve_target_from_active_snapshot(req)
            assert len(target.identities) == 2
            providers = {i.provider for i in target.identities}
            assert providers == {"openai", "xai"}

    @pytest.mark.parametrize("alias_name", ["reference-parent", "reference-branch"])
    def test_alias_reference_expansion_resolves_concrete_candidates(
        self,
        fresh_manager,
        alias_name,
    ):
        leaf_candidate = _make_candidate(
            provider="xai",
            model="grok-reference-target",
            route_family="codex_xai_responses",
        )
        snapshot = RoutingSnapshot(
            aliases={
                "reference-parent": RoutingAlias(
                    name="reference-parent",
                    distribution_strategy=None,
                    candidates=(
                        AliasReference(
                            alias_name="reference-branch",
                            priority=90,
                            weight=1.0,
                        ),
                    ),
                ),
                "reference-branch": RoutingAlias(
                    name="reference-branch",
                    distribution_strategy="proportional",
                    candidates=(
                        AliasReference(
                            alias_name="reference-leaf",
                            priority=100,
                            weight=1.0,
                        ),
                    ),
                ),
                "reference-leaf": RoutingAlias(
                    name="reference-leaf",
                    distribution_strategy=None,
                    candidates=(leaf_candidate,),
                ),
            },
            config_epoch=1,
            config_hash="alias-reference-hash",
            config_version="alias-ref",
        )
        req = CooldownClearRequest(alias=alias_name, ingress="codex")

        with patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot):
            target = _resolve_target_from_active_snapshot(req)

        assert target.canonical_aliases == (alias_name,)
        assert target.target_description == f"alias:{alias_name}"
        assert [
            (identity.provider, identity.model, identity.route_family)
            for identity in target.identities
        ] == [
            (
                "xai",
                "grok-reference-target",
                "codex_xai_responses",
            )
        ]

    def test_exact_resolution_success(self, fresh_manager):
        snapshot = _make_snapshot(provider="openai", model="gpt-4o")
        req = CooldownClearRequest(provider="openai", model="gpt-4o", ingress="codex")
        with patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot):
            target = _resolve_target_from_active_snapshot(req)
            assert target.family == "codex"
            assert "exact:openai/gpt-4o" in target.target_description

    def test_exact_ambiguity_across_route_families_rejected(self, fresh_manager):
        """Exact provider/model matching multiple identities is rejected."""
        cand2 = _make_candidate(
            provider="openai",
            model="gpt-4o",
            route_family="codex_openai_chat",
        )
        snapshot = _make_snapshot(extra_candidates=[cand2])
        req = CooldownClearRequest(provider="openai", model="gpt-4o", ingress="codex")
        with patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot):
            with pytest.raises(HTTPException) as exc_info:
                _resolve_target_from_active_snapshot(req)
            assert exc_info.value.status_code == 400
            assert exc_info.value.detail["error"] == "ambiguous_target"

    def test_anthropic_projection_uses_anthropic_route_family(self, fresh_manager):
        """Anthropic ingress uses anthropic_route_family, not codex route_family."""
        snapshot = _make_snapshot(
            provider="anthropic",
            model="claude-3-opus",
            route_family="codex_openai_responses",
            anthropic_route_family="anthropic_native_messages",
        )
        req = CooldownClearRequest(alias="test-alias", ingress="anthropic")
        with patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot):
            target = _resolve_target_from_active_snapshot(req)
            assert target.family == "anthropic"
            assert target.identities[0].route_family == "anthropic_native_messages"

    def test_anthropic_projection_missing_fails_closed(self, fresh_manager):
        """Anthropic ingress with no anthropic_route_family fails closed."""
        snapshot = _make_snapshot(
            provider="openai",
            model="gpt-4o",
            route_family="codex_openai_responses",
            anthropic_route_family=None,
        )
        req = CooldownClearRequest(alias="test-alias", ingress="anthropic")
        with patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot):
            with pytest.raises(HTTPException) as exc_info:
                _resolve_target_from_active_snapshot(req)
            assert exc_info.value.status_code == 503
            assert exc_info.value.detail["error"] == "anthropic_projection_unavailable"

    def test_codex_ingress_uses_primary_route_family(self, fresh_manager):
        """Codex ingress uses the primary route_family, not anthropic."""
        snapshot = _make_snapshot(
            provider="openai",
            model="gpt-4o",
            route_family="codex_openai_responses",
            anthropic_route_family="anthropic_openai_responses_adapter",
        )
        req = CooldownClearRequest(alias="test-alias", ingress="codex")
        with patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot):
            target = _resolve_target_from_active_snapshot(req)
            assert target.identities[0].route_family == "codex_openai_responses"

    def test_no_raw_keys_in_resolution_error(self, fresh_manager):
        snapshot = _make_snapshot()
        req = CooldownClearRequest(alias="nonexistent", ingress="codex")
        with patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot):
            with pytest.raises(HTTPException) as exc_info:
                _resolve_target_from_active_snapshot(req)
            detail_str = str(exc_info.value.detail)
            assert "aawm:alias-routing" not in detail_str
            assert "sha256" not in detail_str


# ---------------------------------------------------------------------------
# Fail-closed tests (Acceptance #3)
# ---------------------------------------------------------------------------


class TestFailClosed:
    @pytest.mark.asyncio
    async def test_missing_dual_cache_fails_closed(self, fresh_manager):
        """Missing DualCache must fail closed, never return not_active."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache", return_value=None),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            assert exc_info.value.status_code == 503
            assert exc_info.value.detail["error"] == "cache_unavailable"

    @pytest.mark.asyncio
    async def test_redis_inspection_error_fails_closed(self, fresh_manager):
        """Redis identity inspection error must fail closed."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                side_effect=RuntimeError("redis connection refused"),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            assert exc_info.value.status_code == 503
            assert exc_info.value.detail["error"] == "inspection_failed"


# ---------------------------------------------------------------------------
# Idempotent not_active tests (Acceptance #3)
# ---------------------------------------------------------------------------


class TestIdempotentNotActive:
    @pytest.mark.asyncio
    async def test_not_active_after_authoritative_absence(self, fresh_manager):
        """not_active only after both local and durable prove absence."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=absent_inspection,
            ),
        ):
            result = await handle_cooldown_clear(request, admin)
            assert result["result"] == "not_active"
            assert result["keys_cleared"] == 0
            assert result["family"] == "codex"

    @pytest.mark.asyncio
    async def test_not_active_is_idempotent(self, fresh_manager):
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=absent_inspection,
            ),
        ):
            result1 = await handle_cooldown_clear(request, admin)
            result2 = await handle_cooldown_clear(request, admin)
            assert result1["result"] == "not_active"
            assert result2["result"] == "not_active"


# ---------------------------------------------------------------------------
# Successful clear tests
# ---------------------------------------------------------------------------


class TestSuccessfulClear:
    @pytest.mark.asyncio
    async def test_clear_with_active_local_cooldown(self, fresh_manager):
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        _seed_lane_keys(fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[cd_key])

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({cd_key}), cardinality=1
        )
        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                side_effect=[active_inspection, active_inspection, active_inspection, active_inspection, absent_inspection],
            ),
            patch(
                f"{_CLEAR_MOD}.clear_cooldown_transaction",
                new_callable=AsyncMock,
                return_value=_make_txn_result(),
            ),
            patch(
                f"{_CLEAR_MOD}.verify_aawm_alias_routing_durable_absence",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await handle_cooldown_clear(request, admin)
            assert result["result"] == "cleared"
            assert result["keys_cleared"] == 1
            assert result["affinity_preserved"] is True
            assert cd_key not in fresh_manager.codex.cooldown_until_monotonic_by_key


# ---------------------------------------------------------------------------
# Unrelated state preservation tests
# ---------------------------------------------------------------------------


class TestUnrelatedStatePreservation:
    @pytest.mark.asyncio
    async def test_affinity_preserved_after_clear(self, fresh_manager):
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        _seed_lane_keys(fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[cd_key])

        # Seed unrelated affinity + cooldown.
        fresh_manager.codex.session_affinity_by_key["session-123"] = {
            "provider": "openai",
            "model": "gpt-4o",
            "route_family": "codex_openai_responses",
            "expires_at_monotonic": time.monotonic() + 3600.0,
        }
        fresh_manager.codex.cooldown_until_monotonic_by_key["unrelated:key"] = (
            time.monotonic() + 600.0
        )

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({cd_key}), cardinality=1
        )
        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                side_effect=[active_inspection, active_inspection, active_inspection, active_inspection, absent_inspection],
            ),
            patch(
                f"{_CLEAR_MOD}.clear_cooldown_transaction",
                new_callable=AsyncMock,
                return_value=_make_txn_result(),
            ),
            patch(
                f"{_CLEAR_MOD}.verify_aawm_alias_routing_durable_absence",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await handle_cooldown_clear(request, admin)
            assert result["result"] == "cleared"
            assert result["affinity_preserved"] is True

        assert "session-123" in fresh_manager.codex.session_affinity_by_key
        assert "unrelated:key" in fresh_manager.codex.cooldown_until_monotonic_by_key
        assert cd_key not in fresh_manager.codex.cooldown_until_monotonic_by_key


# ---------------------------------------------------------------------------
# Reservation sequencing tests (Acceptance #4)
# ---------------------------------------------------------------------------


class TestReservationSequencing:
    @pytest.mark.asyncio
    async def test_reservation_completed_on_success(self, fresh_manager):
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        _seed_lane_keys(fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[cd_key])

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({cd_key}), cardinality=1
        )
        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                side_effect=[active_inspection, active_inspection, active_inspection, active_inspection, absent_inspection],
            ),
            patch(
                f"{_CLEAR_MOD}.clear_cooldown_transaction",
                new_callable=AsyncMock,
                return_value=_make_txn_result(),
            ),
            patch(
                f"{_CLEAR_MOD}.verify_aawm_alias_routing_durable_absence",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await handle_cooldown_clear(request, admin)
            assert result["result"] == "cleared"

        # No active reservations should remain.
        registry = fresh_manager.publication_intents
        assert registry.get_clear_reservation("codex", cd_key) is None

    @pytest.mark.asyncio
    async def test_reservation_completed_on_failure(self, fresh_manager):
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        _seed_lane_keys(fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[cd_key])

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({cd_key}), cardinality=1
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=active_inspection,
            ),
            patch(
                f"{_CLEAR_MOD}.clear_cooldown_transaction",
                new_callable=AsyncMock,
                side_effect=RuntimeError("redis down"),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            assert exc_info.value.status_code == 503

        registry = fresh_manager.publication_intents
        assert registry.get_clear_reservation("codex", cd_key) is None

    @pytest.mark.asyncio
    async def test_endpoint_lock_serializes_execution(self, fresh_manager):
        """Endpoint-owned family lock serializes clear executions."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _get_endpoint_family_lock,
        )

        # Verify the lock is created per-family and reused.
        lock1 = await _get_endpoint_family_lock("codex")
        lock2 = await _get_endpoint_family_lock("codex")
        lock3 = await _get_endpoint_family_lock("anthropic")
        assert lock1 is lock2  # Same family -> same lock
        assert lock1 is not lock3  # Different family -> different lock

    @pytest.mark.asyncio
    async def test_sequential_clears_both_succeed(self, fresh_manager):
        """Two sequential clears: first clears, second finds not_active."""
        snapshot = _make_snapshot()
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        _seed_lane_keys(fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[cd_key])

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({cd_key}), cardinality=1
        )
        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                side_effect=[
                    # Req1: hydration, post-drain, prior-state, durable-clear, postcondition
                    active_inspection, active_inspection, active_inspection, active_inspection, absent_inspection,
                    # Req2: hydration, post-drain, prior-state (all absent)
                    absent_inspection, absent_inspection, absent_inspection,
                ],
            ),
            patch(
                f"{_CLEAR_MOD}.clear_cooldown_transaction",
                new_callable=AsyncMock,
                return_value=_make_txn_result(),
            ),
            patch(
                f"{_CLEAR_MOD}.verify_aawm_alias_routing_durable_absence",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            req1 = _make_request({"alias": "test-alias", "ingress": "codex"})
            r1 = await handle_cooldown_clear(req1, admin)
            assert r1["result"] == "cleared"

            req2 = _make_request({"alias": "test-alias", "ingress": "codex"})
            r2 = await handle_cooldown_clear(req2, admin)
            assert r2["result"] == "not_active"


# ---------------------------------------------------------------------------
# Durable failure mode tests
# ---------------------------------------------------------------------------


class TestDurableFailureModes:
    @pytest.mark.asyncio
    async def test_membership_drift_returns_409(self, fresh_manager):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
            MembershipDriftError,
        )

        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        _seed_lane_keys(fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[cd_key])

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({cd_key}), cardinality=1
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=active_inspection,
            ),
            patch(
                f"{_CLEAR_MOD}.clear_cooldown_transaction",
                new_callable=AsyncMock,
                side_effect=MembershipDriftError(
                    phase="PREPARED",
                    family="codex",
                    transaction_id_prefix="txn",
                    identity_prefix="id",
                    key_count=1,
                    exception_classes=(),
                ),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            assert exc_info.value.status_code == 409
            assert exc_info.value.detail["error"] == "membership_drift"

    @pytest.mark.asyncio
    async def test_indeterminate_clear_returns_503(self, fresh_manager):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
            ClearIndeterminateError,
        )

        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        _seed_lane_keys(fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[cd_key])

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({cd_key}), cardinality=1
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=active_inspection,
            ),
            patch(
                f"{_CLEAR_MOD}.clear_cooldown_transaction",
                new_callable=AsyncMock,
                side_effect=ClearIndeterminateError(
                    phase="PREPARED",
                    family="codex",
                    transaction_id_prefix="txn",
                    identity_prefix="id",
                    key_count=1,
                    exception_classes=(),
                ),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            assert exc_info.value.status_code == 503
            assert exc_info.value.detail["error"] == "indeterminate_clear"


# ---------------------------------------------------------------------------
# Audit event tests (Acceptance #6)
# ---------------------------------------------------------------------------


class TestAuditEvents:
    def test_audit_event_contains_required_fields(self):
        target = _ResolvedTarget(
            family="codex",
            target_description="alias:test-alias",
            ingress="codex",
        )
        with patch(f"{_CLEAR_MOD}.logger") as mock_logger:
            _emit_audit_event(
                event_type="success",
                target=target,
                result="cleared",
                prior_state_source="memory",
                bounded_remaining_ttl_seconds=120.5,
            )
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            payload = call_args[0][1]
            assert payload["event"] == "aawm_cooldown_clear_success"
            assert payload["target_description"] == "alias:test-alias"
            assert payload["family"] == "codex"
            assert payload["ingress"] == "codex"
            assert payload["result"] == "cleared"
            assert payload["prior_state_source"] == "memory"
            assert payload["bounded_remaining_ttl_seconds"] == 120.5
            assert "environment" in payload
            assert "namespace" in payload

    def test_audit_event_never_exposes_secrets(self):
        target = _ResolvedTarget(
            family="codex",
            target_description="alias:test-alias",
            ingress="codex",
        )
        with patch(f"{_CLEAR_MOD}.logger") as mock_logger:
            _emit_audit_event(
                event_type="failure",
                target=target,
                result="error",
                error_code="clear_failed",
            )
            call_args = mock_logger.info.call_args
            payload_str = str(call_args)
            assert "sk-" not in payload_str
            assert "Bearer" not in payload_str
            assert "traceback" not in payload_str.lower()

    @pytest.mark.asyncio
    async def test_not_active_emits_audit_event(self, fresh_manager):
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=absent_inspection,
            ),
            patch(f"{_CLEAR_MOD}.logger") as mock_logger,
        ):
            result = await handle_cooldown_clear(request, admin)
            assert result["result"] == "not_active"
            mock_logger.info.assert_called_once()
            payload = mock_logger.info.call_args[0][1]
            assert payload["event"] == "aawm_cooldown_clear_not_active"


# ---------------------------------------------------------------------------
# Response schema tests (Acceptance #6)
# ---------------------------------------------------------------------------


class TestResponseSchema:
    @pytest.mark.asyncio
    async def test_not_active_response_has_required_fields(self, fresh_manager):
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=absent_inspection,
            ),
        ):
            result = await handle_cooldown_clear(request, admin)
            assert "result" in result
            assert "family" in result
            assert "target_description" in result
            assert "ingress" in result
            assert "candidates" in result
            assert "prior_state_source" in result
            assert "bounded_remaining_ttl_seconds" in result
            assert "environment" in result
            assert "namespace" in result
            assert "timestamp_utc" in result
            # No raw keys/hashes/credentials.
            result_str = str(result)
            assert "sha256" not in result_str
            assert "sk-" not in result_str


# ---------------------------------------------------------------------------
# Error redaction tests
# ---------------------------------------------------------------------------


class TestErrorRedaction:
    def test_no_secrets_in_topology_error(self, monkeypatch):
        monkeypatch.delenv("AAWM_ALIAS_ROUTING_COOLDOWN_CLEAR_SINGLE_WORKER", raising=False)
        with pytest.raises(HTTPException) as exc_info:
            _check_topology_gate()
        detail_str = str(exc_info.value.detail)
        assert "sk-" not in detail_str
        assert "password" not in detail_str.lower()
        assert "redis://" not in detail_str

    def test_no_secrets_in_auth_error(self):
        from litellm.proxy._types import LitellmUserRoles

        mock = MagicMock()
        mock.user_role = LitellmUserRoles.INTERNAL_USER
        with pytest.raises(HTTPException) as exc_info:
            _check_admin_auth(mock)
        detail_str = str(exc_info.value.detail)
        assert "sk-" not in detail_str
        assert "Bearer" not in detail_str
        assert "authorization" not in detail_str.lower()


# ---------------------------------------------------------------------------
# Stale-read prevention tests
# ---------------------------------------------------------------------------


class TestStaleReadPrevention:
    @pytest.mark.asyncio
    async def test_generation_bumped_after_clear(self, fresh_manager):
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        _seed_lane_keys(fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[cd_key])

        gen_before = fresh_manager.codex.get_generation(cd_key)

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({cd_key}), cardinality=1
        )
        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                side_effect=[active_inspection, active_inspection, active_inspection, active_inspection, absent_inspection],
            ),
            patch(
                f"{_CLEAR_MOD}.clear_cooldown_transaction",
                new_callable=AsyncMock,
                return_value=_make_txn_result(),
            ),
            patch(
                f"{_CLEAR_MOD}.verify_aawm_alias_routing_durable_absence",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await handle_cooldown_clear(request, admin)
            assert result["result"] == "cleared"

        gen_after = fresh_manager.codex.get_generation(cd_key)
        assert gen_after > gen_before


# ---------------------------------------------------------------------------
# TOS boundary: no provider traffic
# ---------------------------------------------------------------------------


class TestTOSBoundary:
    @pytest.mark.asyncio
    async def test_no_provider_traffic_sent(self, fresh_manager):
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=absent_inspection,
            ),
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_httpx,
        ):
            result = await handle_cooldown_clear(request, admin)
            assert result["result"] == "not_active"
            mock_httpx.assert_not_called()


# ---------------------------------------------------------------------------
# Defect 2: Durable-only restart/index-loss tests
# ---------------------------------------------------------------------------


class TestDurableOnlyRestartIndexLoss:
    @pytest.mark.asyncio
    async def test_durable_only_hydration_when_local_index_empty(self, fresh_manager):
        """When local lane_identity_index is empty (restart/index loss),
        lane keys are hydrated from the authoritative durable identity set."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        # Do NOT seed local index -- simulate restart/index loss.
        # Only durable has the membership.

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({cd_key}), cardinality=1
        )
        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                side_effect=[
                    active_inspection,   # hydration
                    active_inspection,   # post-drain rehydration
                    active_inspection,   # prior-state
                    active_inspection,   # durable-clear
                    absent_inspection,   # postcondition
                ],
            ),
            patch(
                f"{_CLEAR_MOD}.clear_cooldown_transaction",
                new_callable=AsyncMock,
                return_value=_make_txn_result(),
            ),
            patch(
                f"{_CLEAR_MOD}.verify_aawm_alias_routing_durable_absence",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await handle_cooldown_clear(request, admin)
            assert result["result"] == "cleared"
            assert result["keys_cleared"] == 1
            assert result["members_removed"] == 1

    @pytest.mark.asyncio
    async def test_durable_only_not_active_when_no_durable_members(self, fresh_manager):
        """When local index is empty AND durable has no members,
        result is not_active (never cleared with zero transactions)."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=absent_inspection,
            ),
        ):
            result = await handle_cooldown_clear(request, admin)
            assert result["result"] == "not_active"
            assert result["keys_cleared"] == 0

    @pytest.mark.asyncio
    async def test_durable_only_never_returns_cleared_with_zero_txn_calls(self, fresh_manager):
        """Durable members exist but local index empty: must call
        clear_cooldown_transaction, never return cleared with zero calls."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({cd_key}), cardinality=1
        )
        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )
        mock_txn = AsyncMock(return_value=_make_txn_result())

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                side_effect=[
                    active_inspection,   # hydration
                    active_inspection,   # post-drain rehydration
                    active_inspection,   # prior-state
                    active_inspection,   # durable-clear
                    absent_inspection,   # postcondition
                ],
            ),
            patch(f"{_CLEAR_MOD}.clear_cooldown_transaction", mock_txn),
            patch(
                f"{_CLEAR_MOD}.verify_aawm_alias_routing_durable_absence",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await handle_cooldown_clear(request, admin)
            assert result["result"] == "cleared"
            # Transaction MUST have been called at least once.
            assert mock_txn.call_count >= 1

    @pytest.mark.asyncio
    async def test_unrelated_identity_not_cleared(self, fresh_manager):
        """Durable hydration uses exact identity members; unrelated
        identities cannot be cleared."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        # Durable returns only the exact member for THIS identity.
        own_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        unrelated_key = "hOTHER:xai:grok-3:chatgpt-account:acct99"

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({own_key}), cardinality=1
        )
        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                side_effect=[
                    active_inspection,   # hydration
                    active_inspection,   # post-drain rehydration
                    active_inspection,   # prior-state
                    active_inspection,   # durable-clear
                    absent_inspection,   # postcondition
                ],
            ),
            patch(
                f"{_CLEAR_MOD}.clear_cooldown_transaction",
                new_callable=AsyncMock,
                return_value=_make_txn_result(),
            ),
            patch(
                f"{_CLEAR_MOD}.verify_aawm_alias_routing_durable_absence",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await handle_cooldown_clear(request, admin)
            assert result["result"] == "cleared"
            # Unrelated key must NOT appear in cleared keys.
            assert unrelated_key not in str(result)


# ---------------------------------------------------------------------------
# Defect 3: Deterministic serialization tests
# ---------------------------------------------------------------------------


class TestDeterministicSerialization:
    @pytest.mark.asyncio
    async def test_two_concurrent_clears_one_cleared_one_not_active(self, fresh_manager):
        """Two concurrent valid clears: one clears, the follower observes
        authoritative not_active.  Never two 'cleared' claims."""
        snapshot = _make_snapshot()
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        _seed_lane_keys(fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[cd_key])

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({cd_key}), cardinality=1
        )
        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        # Track call count to provide correct side_effects.
        call_count = 0

        async def inspect_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            # Calls 1-3: first request (hydration, prior-state, durable-clear)
            # Calls 4+: postcondition + second request (all absent)
            if call_count <= 4:
                return active_inspection
            return absent_inspection

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                side_effect=inspect_side_effect,
            ),
            patch(
                f"{_CLEAR_MOD}.clear_cooldown_transaction",
                new_callable=AsyncMock,
                return_value=_make_txn_result(),
            ),
            patch(
                f"{_CLEAR_MOD}.verify_aawm_alias_routing_durable_absence",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            req1 = _make_request({"alias": "test-alias", "ingress": "codex"})
            req2 = _make_request({"alias": "test-alias", "ingress": "codex"})

            r1, r2 = await asyncio.gather(
                handle_cooldown_clear(req1, admin),
                handle_cooldown_clear(req2, admin),
            )

            results = {r1["result"], r2["result"]}
            # Exactly one cleared, one not_active.
            assert "cleared" in results
            assert "not_active" in results
            # Never two cleared.
            cleared_count = sum(1 for r in (r1, r2) if r["result"] == "cleared")
            assert cleared_count == 1

    @pytest.mark.asyncio
    async def test_reservation_established_before_inspection(self, fresh_manager):
        """Reservation is established BEFORE prior-state inspection,
        so an active prior publication cannot yield not_active without
        the reservation having been established first."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        # Register lane keys in index but do NOT seed cooldown state.
        fresh_manager.lane_identity_index.register_batch(
            identity_hash=id_hash,
            lane_keys=[cd_key],
        )

        reservation_seen_before_inspection = False

        async def tracking_inspect(**kwargs):
            nonlocal reservation_seen_before_inspection
            # Check if reservation exists at inspection time.
            res = fresh_manager.publication_intents.get_clear_reservation("codex", cd_key)
            if res is not None:
                reservation_seen_before_inspection = True
            return _make_identity_inspection(
                exists=False, members=frozenset(), cardinality=0
            )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                side_effect=tracking_inspect,
            ),
            patch(
                f"{_CLEAR_MOD}.verify_aawm_alias_routing_durable_absence",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await handle_cooldown_clear(request, admin)
            assert result["result"] == "not_active"
            # Reservation MUST have been visible during inspection.
            assert reservation_seen_before_inspection is True


# ---------------------------------------------------------------------------
# Defect 4: Negative/evidence state absence tests
# ---------------------------------------------------------------------------


class TestNegativeEvidenceStateAbsence:
    @pytest.mark.asyncio
    async def test_not_active_requires_negative_cache_absent(self, fresh_manager):
        """not_active must NOT be returned when negative-cache entries exist."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        _seed_lane_keys(fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[cd_key])

        # Seed negative cache (no positive cooldown).
        fresh_manager.codex.cooldown_negative_until_monotonic_by_key[cd_key] = (
            time.monotonic() + 300.0
        )

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({cd_key}), cardinality=1
        )
        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                side_effect=[active_inspection, active_inspection, active_inspection, active_inspection, absent_inspection],
            ),
            patch(
                f"{_CLEAR_MOD}.clear_cooldown_transaction",
                new_callable=AsyncMock,
                return_value=_make_txn_result(),
            ),
            patch(
                f"{_CLEAR_MOD}.verify_aawm_alias_routing_durable_absence",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await handle_cooldown_clear(request, admin)
            # Must be cleared (negative cache counts as active state).
            assert result["result"] == "cleared"
            # Negative cache must be gone.
            assert cd_key not in fresh_manager.codex.cooldown_negative_until_monotonic_by_key

    @pytest.mark.asyncio
    async def test_not_active_requires_evidence_absent(self, fresh_manager):
        """not_active must NOT be returned when evidence events exist."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        _seed_lane_keys(fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[cd_key])

        # Seed evidence events (no positive cooldown, no negative cache).
        fresh_manager.codex.evidence_events_by_key[cd_key] = [time.monotonic()]

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({cd_key}), cardinality=1
        )
        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                side_effect=[active_inspection, active_inspection, active_inspection, active_inspection, absent_inspection],
            ),
            patch(
                f"{_CLEAR_MOD}.clear_cooldown_transaction",
                new_callable=AsyncMock,
                return_value=_make_txn_result(),
            ),
            patch(
                f"{_CLEAR_MOD}.verify_aawm_alias_routing_durable_absence",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await handle_cooldown_clear(request, admin)
            assert result["result"] == "cleared"
            assert cd_key not in fresh_manager.codex.evidence_events_by_key

    @pytest.mark.asyncio
    async def test_clear_removes_all_cooldown_derived_state(self, fresh_manager):
        """After clear, ALL cooldown-derived state is absent: positive,
        negative, evidence, and local index."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        _seed_lane_keys(fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[cd_key])

        # Seed ALL cooldown-derived state.
        fresh_manager.codex.cooldown_negative_until_monotonic_by_key[cd_key] = (
            time.monotonic() + 300.0
        )
        fresh_manager.codex.evidence_events_by_key[cd_key] = [time.monotonic()]

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({cd_key}), cardinality=1
        )
        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                side_effect=[active_inspection, active_inspection, active_inspection, active_inspection, absent_inspection],
            ),
            patch(
                f"{_CLEAR_MOD}.clear_cooldown_transaction",
                new_callable=AsyncMock,
                return_value=_make_txn_result(),
            ),
            patch(
                f"{_CLEAR_MOD}.verify_aawm_alias_routing_durable_absence",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await handle_cooldown_clear(request, admin)
            assert result["result"] == "cleared"

        # Strict postcondition: ALL absent.
        assert cd_key not in fresh_manager.codex.cooldown_until_monotonic_by_key
        assert cd_key not in fresh_manager.codex.cooldown_negative_until_monotonic_by_key
        assert cd_key not in fresh_manager.codex.evidence_events_by_key
        assert fresh_manager.lane_identity_index.lanes_for(id_hash) == frozenset()


# ---------------------------------------------------------------------------
# Defect 5: Environment field tests
# ---------------------------------------------------------------------------


class TestEnvironmentField:
    @pytest.mark.asyncio
    async def test_environment_uses_aawm_litellm_environment(self, fresh_manager, monkeypatch):
        """Environment field uses AAWM_LITELLM_ENVIRONMENT as primary source."""
        monkeypatch.setenv("AAWM_LITELLM_ENVIRONMENT", "litellm-dev")
        monkeypatch.delenv("LITELLM_ENVIRONMENT", raising=False)

        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=absent_inspection,
            ),
        ):
            result = await handle_cooldown_clear(request, admin)
            assert result["environment"] == "litellm-dev"

    @pytest.mark.asyncio
    async def test_environment_falls_back_to_litellm_environment(self, fresh_manager, monkeypatch):
        """Environment falls back to LITELLM_ENVIRONMENT when AAWM var absent."""
        monkeypatch.delenv("AAWM_LITELLM_ENVIRONMENT", raising=False)
        monkeypatch.setenv("LITELLM_ENVIRONMENT", "staging")

        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=absent_inspection,
            ),
        ):
            result = await handle_cooldown_clear(request, admin)
            assert result["environment"] == "staging"

    @pytest.mark.asyncio
    async def test_environment_unknown_when_neither_set(self, fresh_manager, monkeypatch):
        """Environment is 'unknown' when neither env var is set."""
        monkeypatch.delenv("AAWM_LITELLM_ENVIRONMENT", raising=False)
        monkeypatch.delenv("LITELLM_ENVIRONMENT", raising=False)

        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=absent_inspection,
            ),
        ):
            result = await handle_cooldown_clear(request, admin)
            assert result["environment"] == "unknown"

    @pytest.mark.asyncio
    async def test_audit_event_uses_correct_environment(self, fresh_manager, monkeypatch):
        """Audit events use the same environment resolution."""
        monkeypatch.setenv("AAWM_LITELLM_ENVIRONMENT", "prod")

        target = _ResolvedTarget(
            family="codex",
            target_description="alias:test-alias",
            ingress="codex",
        )
        with patch(f"{_CLEAR_MOD}.logger") as mock_logger:
            _emit_audit_event(
                event_type="success",
                target=target,
                result="cleared",
                prior_state_source="memory",
            )
            payload = mock_logger.info.call_args[0][1]
            assert payload["environment"] == "prod"


# ---------------------------------------------------------------------------
# Finding 1: Partial local-index recovery (union durable + local)
# ---------------------------------------------------------------------------


class TestPartialLocalIndexRecovery:
    @pytest.mark.asyncio
    async def test_partial_index_unions_durable_only_lane(self, fresh_manager):
        """Local index has ONE lane; durable has a SECOND durable-only lane.
        The clear must cover the full exact durable+local union set."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        local_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        durable_only_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct2"

        # Seed ONLY the local lane in the index (partial index).
        fresh_manager.lane_identity_index.register_batch(
            identity_hash=id_hash, lane_keys=[local_key]
        )
        # Seed local cooldown for the local lane so prior-state is active.
        fresh_manager.codex.cooldown_until_monotonic_by_key[local_key] = (
            time.monotonic() + 300.0
        )

        # Durable authoritatively knows BOTH lanes (local + durable-only).
        union_members = frozenset({local_key, durable_only_key})
        active_inspection = _make_identity_inspection(
            exists=True, members=union_members, cardinality=2
        )
        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        mock_txn = AsyncMock(return_value=_make_txn_result(keys_deleted=2, members_removed=2))

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                side_effect=[
                    active_inspection,   # hydration (union)
                    active_inspection,   # post-drain rehydration
                    active_inspection,   # prior-state
                    active_inspection,   # durable-clear inspect
                    absent_inspection,   # postcondition
                ],
            ),
            patch(f"{_CLEAR_MOD}.clear_cooldown_transaction", mock_txn),
            patch(
                f"{_CLEAR_MOD}.verify_aawm_alias_routing_durable_absence",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await handle_cooldown_clear(request, admin)
            assert result["result"] == "cleared"
            # Both lanes cleared: the durable-only lane must be included.
            assert result["keys_cleared"] == 2
            assert result["members_removed"] == 2

        # The durable-only lane must have been part of the clear transaction.
        txn_keys = mock_txn.call_args.kwargs["cooldown_keys"]
        assert set(txn_keys) == {local_key, durable_only_key}
        # Local index fully cleared for the identity.
        assert fresh_manager.lane_identity_index.lanes_for(id_hash) == frozenset()


# ---------------------------------------------------------------------------
# Finding 2: Identity reservation timing (blocks first publication)
# ---------------------------------------------------------------------------


class TestIdentityReservationTiming:
    @pytest.mark.asyncio
    async def test_reservation_exists_during_first_identity_inspection(self, fresh_manager):
        """An identity-scoped reservation exists during the FIRST Redis
        identity inspection (hydration) and blocks a first-ever publication."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
            ClaimOutcome,
        )

        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"

        reservation_seen_at_first_inspection = False
        first_publication_blocked = False
        call_count = 0

        async def tracking_inspect(**kwargs):
            nonlocal reservation_seen_at_first_inspection, first_publication_blocked
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # FIRST Redis identity inspection (hydration): the
                # identity-scoped reservation must already be active.
                res = fresh_manager.publication_intents.get_clear_reservation_by_identity(
                    "codex", id_hash
                )
                if res is not None:
                    reservation_seen_at_first_inspection = True
                    # And it must block a first-ever publication for this
                    # identity even though no cooldown_key is known yet.
                    claim = fresh_manager.publication_intents.claim_publication_or_wait(
                        alias_family="codex",
                        cooldown_keys=frozenset({cd_key}),
                        identity_hash=id_hash,
                    )
                    if claim.outcome is ClaimOutcome.BLOCKED_BY_CLEAR:
                        first_publication_blocked = True
            return _make_identity_inspection(
                exists=False, members=frozenset(), cardinality=0
            )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                side_effect=tracking_inspect,
            ),
            patch(
                f"{_CLEAR_MOD}.verify_aawm_alias_routing_durable_absence",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await handle_cooldown_clear(request, admin)
            assert result["result"] == "not_active"
            assert reservation_seen_at_first_inspection is True
            assert first_publication_blocked is True


# ---------------------------------------------------------------------------
# Finding 3: not_active per-key durable absence proof
# ---------------------------------------------------------------------------


class TestNotActivePerKeyProof:
    @pytest.mark.asyncio
    async def test_not_active_blocked_when_redis_key_still_present(self, fresh_manager):
        """Identity-set says absent but a stale durable cooldown KEY is still
        present: not_active must fail closed, never return not_active."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        # Local index knows the lane, but no local cooldown state.
        fresh_manager.lane_identity_index.register_batch(
            identity_hash=id_hash, lane_keys=[cd_key]
        )

        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=absent_inspection,
            ),
            # Stale DualCache/Redis: the per-key verifier reports the key
            # is STILL present.
            patch(
                f"{_CLEAR_MOD}.verify_aawm_alias_routing_durable_absence",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            assert exc_info.value.status_code == 503
            assert exc_info.value.detail["error"] == "absence_verification_failed"

    @pytest.mark.asyncio
    async def test_not_active_fails_closed_on_verifier_uncertainty(self, fresh_manager):
        """Verifier uncertainty (exception) fails closed before not_active."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        fresh_manager.lane_identity_index.register_batch(
            identity_hash=id_hash, lane_keys=[cd_key]
        )

        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=absent_inspection,
            ),
            patch(
                f"{_CLEAR_MOD}.verify_aawm_alias_routing_durable_absence",
                new_callable=AsyncMock,
                side_effect=RuntimeError("redis timeout"),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            assert exc_info.value.status_code == 503
            assert exc_info.value.detail["error"] == "absence_verification_failed"


# ---------------------------------------------------------------------------
# Finding 4: Codex classification-marker evidence before key state
# ---------------------------------------------------------------------------


class TestCodexFailureMarkerEvidence:
    def test_marker_evidence_before_key_state_detected_and_cleared(self, fresh_manager):
        """Marker-tier Codex evidence accumulates in the alias gate's family
        evidence map BEFORE any _key_state entry exists.  inspect_cooldown_absence
        must detect it (not classify it absent) and clear must remove it."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
            inspect_cooldown_absence,
        )

        cd_key = "hcfg:openai:gpt-4o:chatgpt-account:acct1"
        canonical_alias = "test-alias"
        gate = fresh_manager.codex_failure_evidence_gate.gate_for_alias(
            canonical_alias=canonical_alias,
            create=True,
        )
        assert gate is not None
        # Seed marker evidence ONLY in the family evidence map -- no _key_state.
        gate._family_state.evidence_events_by_key[cd_key] = [time.monotonic()]
        assert cd_key not in gate._key_state

        inspection = inspect_cooldown_absence(
            fresh_manager,
            alias_family="codex",
            canonical_aliases=[canonical_alias],
            cooldown_key=cd_key,
        )
        assert inspection.exists is True
        assert inspection.codex_failure_evidence_present is True

        # Clear must remove the marker evidence so it cannot survive.
        fresh_manager.clear_cooldown_state(
            alias_family="codex",
            canonical_aliases=[canonical_alias],
            cooldown_keys=[cd_key],
        )

        after = inspect_cooldown_absence(
            fresh_manager,
            alias_family="codex",
            canonical_aliases=[canonical_alias],
            cooldown_key=cd_key,
        )
        assert after.exists is False
        assert after.codex_failure_evidence_present is False
        assert cd_key not in gate._family_state.evidence_events_by_key


# ---------------------------------------------------------------------------
# Fix 1: First-ever publication race regression
# ---------------------------------------------------------------------------


class TestFirstEverPublicationRace:
    """Deterministic regression: pause first local/durable inspection,
    attempt first-ever publication for that identity, assert it is
    blocked/waits, then endpoint cannot return not_active while intent
    remains active."""

    @pytest.mark.asyncio
    async def test_first_ever_publication_blocked_during_local_inspection(
        self, fresh_manager
    ):
        """Modeled on the default validator /tmp probe: the identity-scoped
        reservation is created BEFORE any local/durable inspection, so a
        first-ever publication attempt during local index hydration is
        blocked by the reservation and cannot become leader."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
            ClaimOutcome,
        )

        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"

        publication_blocked_at_local_inspection = False
        reservation_active_at_local_inspection = False
        local_inspection_happened = False

        original_hydrate = _hydrate_from_local_index

        def pausing_hydrate(target, state_mgr):
            """Pause at local index hydration and attempt first-ever publication."""
            nonlocal publication_blocked_at_local_inspection
            nonlocal reservation_active_at_local_inspection
            nonlocal local_inspection_happened
            local_inspection_happened = True

            # At this point the reservation MUST already exist.
            res = state_mgr.publication_intents.get_clear_reservation_by_identity(
                "codex", id_hash
            )
            if res is not None:
                reservation_active_at_local_inspection = True
                # Attempt first-ever publication for this identity.
                claim = state_mgr.publication_intents.claim_publication_or_wait(
                    alias_family="codex",
                    cooldown_keys=frozenset({cd_key}),
                    identity_hash=id_hash,
                )
                if claim.outcome is ClaimOutcome.BLOCKED_BY_CLEAR:
                    publication_blocked_at_local_inspection = True

            # Proceed with real hydration.
            original_hydrate(target, state_mgr)

        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=absent_inspection,
            ),
            patch(
                f"{_CLEAR_MOD}.verify_aawm_alias_routing_durable_absence",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                f"{_CLEAR_MOD}._hydrate_from_local_index",
                side_effect=pausing_hydrate,
            ),
        ):
            result = await handle_cooldown_clear(request, admin)
            # Endpoint returns not_active (no active cooldowns), but the
            # first-ever publication was blocked during local inspection.
            assert result["result"] == "not_active"
            assert local_inspection_happened is True
            assert reservation_active_at_local_inspection is True
            assert publication_blocked_at_local_inspection is True

    @pytest.mark.asyncio
    async def test_publication_cannot_become_leader_before_reservation(
        self, fresh_manager
    ):
        """Verify that pure snapshot resolution does NOT read lane index,
        so no publication claim can sneak in between resolution and
        reservation creation."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
            ClaimOutcome,
        )

        snapshot = _make_snapshot()
        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"

        # Before any reservation: publication CAN become leader.
        claim_before = fresh_manager.publication_intents.claim_publication_or_wait(
            alias_family="codex",
            cooldown_keys=frozenset({cd_key}),
            identity_hash=id_hash,
        )
        assert claim_before.outcome is ClaimOutcome.LEADER
        # Clean up the intent.
        fresh_manager.publication_intents.release_claim(claim_before.intent)

        # After reservation (as endpoint creates it): publication is blocked.
        fresh_manager.publication_intents.create_clear_reservation(
            alias_family="codex",
            identity_hashes=frozenset({id_hash}),
            cooldown_keys=frozenset(),
        )
        claim_after = fresh_manager.publication_intents.claim_publication_or_wait(
            alias_family="codex",
            cooldown_keys=frozenset({cd_key}),
            identity_hash=id_hash,
        )
        assert claim_after.outcome is ClaimOutcome.BLOCKED_BY_CLEAR


# ---------------------------------------------------------------------------
# Fix 2: Audit events for every failure/conflict phase
# ---------------------------------------------------------------------------


class TestAuditEventsEveryPhase:
    """Each failure phase produces exactly one redacted audit event."""

    @pytest.mark.asyncio
    async def test_schema_failure_emits_one_audit(self, fresh_manager):
        """Schema validation failure emits exactly one audit event."""
        request = _make_request({"alias": "test", "ingress": "codex", "extra": "bad"})
        admin = _make_real_admin_key_dict()

        with (
            patch(f"{_CLEAR_MOD}.logger") as mock_logger,
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            assert exc_info.value.status_code == 400
            assert exc_info.value.detail["error"] == "unexpected_fields"
            # Exactly one audit event.
            assert mock_logger.info.call_count == 1
            payload = mock_logger.info.call_args[0][1]
            assert payload["event"] == "aawm_cooldown_clear_failure"
            assert payload["error_code"] == "unexpected_fields"
            # Redacted: no secrets.
            payload_str = str(payload)
            assert "sk-" not in payload_str

    @pytest.mark.asyncio
    async def test_invalid_body_emits_one_audit(self, fresh_manager):
        """Invalid JSON body emits exactly one audit event."""
        request = MagicMock()
        request.json = AsyncMock(side_effect=ValueError("bad json"))
        admin = _make_real_admin_key_dict()

        with (
            patch(f"{_CLEAR_MOD}.logger") as mock_logger,
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            assert exc_info.value.status_code == 400
            assert mock_logger.info.call_count == 1
            payload = mock_logger.info.call_args[0][1]
            assert payload["error_code"] == "invalid_body"

    @pytest.mark.asyncio
    async def test_auth_failure_emits_one_audit(self, fresh_manager):
        """Auth failure emits exactly one audit event."""
        from litellm.proxy._types import LitellmUserRoles

        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        non_admin = MagicMock()
        non_admin.user_role = LitellmUserRoles.INTERNAL_USER

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch(f"{_CLEAR_MOD}.logger") as mock_logger,
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, non_admin)
            assert exc_info.value.status_code == 403
            assert mock_logger.info.call_count == 1
            payload = mock_logger.info.call_args[0][1]
            assert payload["event"] == "aawm_cooldown_clear_failure"
            assert payload["error_code"] == "forbidden"
            # Redacted.
            payload_str = str(payload)
            assert "sk-" not in payload_str
            assert "Bearer" not in payload_str

    @pytest.mark.asyncio
    async def test_topology_gate_failure_emits_one_audit(self, fresh_manager, monkeypatch):
        """Topology gate failure emits exactly one audit event."""
        monkeypatch.delenv("AAWM_ALIAS_ROUTING_COOLDOWN_CLEAR_SINGLE_WORKER", raising=False)
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.logger") as mock_logger,
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            assert exc_info.value.status_code == 503
            assert mock_logger.info.call_count == 1
            payload = mock_logger.info.call_args[0][1]
            assert payload["error_code"] == "topology_gate_closed"

    @pytest.mark.asyncio
    async def test_resolution_failure_emits_one_audit(self, fresh_manager):
        """Snapshot resolution failure emits exactly one audit event."""
        request = _make_request({"alias": "nonexistent", "ingress": "codex"})
        admin = _make_real_admin_key_dict()
        snapshot = _make_snapshot(alias_name="other")

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.logger") as mock_logger,
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            assert exc_info.value.status_code == 404
            assert mock_logger.info.call_count == 1
            payload = mock_logger.info.call_args[0][1]
            assert payload["error_code"] == "alias_not_found"

    @pytest.mark.asyncio
    async def test_reservation_failure_emits_one_audit(self, fresh_manager):
        """Reservation RuntimeError emits one audit and sanitized 503."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(f"{_CLEAR_MOD}.logger") as mock_logger,
            patch.object(
                fresh_manager.publication_intents,
                "create_clear_reservation",
                side_effect=RuntimeError("registry corrupted"),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            # Sanitized 503 -- no exception text leaked.
            assert exc_info.value.status_code == 503
            assert exc_info.value.detail["error"] == "internal_error"
            assert "registry corrupted" not in str(exc_info.value.detail)
            assert "RuntimeError" not in str(exc_info.value.detail)
            # Exactly one audit event.
            assert mock_logger.info.call_count == 1
            payload = mock_logger.info.call_args[0][1]
            assert payload["event"] == "aawm_cooldown_clear_failure"
            assert payload["error_code"] == "internal_error"
            # Audit must not leak exception internals.
            payload_str = str(payload)
            assert "registry corrupted" not in payload_str
            assert "traceback" not in payload_str.lower()

    @pytest.mark.asyncio
    async def test_unexpected_runtime_error_sanitized_503(self, fresh_manager):
        """Unexpected RuntimeError produces sanitized 503 with no internals."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(f"{_CLEAR_MOD}.logger") as mock_logger,
            patch.object(
                fresh_manager.publication_intents,
                "create_clear_reservation",
                side_effect=RuntimeError("secret internal detail"),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            assert exc_info.value.status_code == 503
            detail_str = str(exc_info.value.detail)
            # No exception text, traceback, vars, or internal detail leaked.
            assert "secret internal detail" not in detail_str
            assert "RuntimeError" not in detail_str
            assert "Traceback" not in detail_str
            # Exactly one audit.
            assert mock_logger.info.call_count == 1
            payload = mock_logger.info.call_args[0][1]
            assert payload["error_code"] == "internal_error"
            assert payload["result"] == "error"
            # Partial target fields are safe strings.
            assert isinstance(payload["target_description"], str)
            assert isinstance(payload["environment"], str)
            assert isinstance(payload["namespace"], str)

    @pytest.mark.asyncio
    async def test_no_duplicate_audit_when_inner_already_emitted(self, fresh_manager):
        """If inner catch already emitted audit, outer Exception handler
        does not emit a second one (duplicate suppression)."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        _seed_lane_keys(fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[cd_key])

        # Hydration fails with HTTPException (inner catch fires, emits audit),
        # then _complete_reservation_if_owned in finally raises RuntimeError,
        # which replaces the HTTPException and hits the outer except Exception.

        def exploding_complete(*args, **kwargs):
            raise RuntimeError("finally block exploded")

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=None,  # Triggers HTTPException in hydration.
            ),
            patch(f"{_CLEAR_MOD}.logger") as mock_logger,
            patch(
                f"{_CLEAR_MOD}._complete_reservation_if_owned",
                side_effect=exploding_complete,
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            # Outer handler converts to sanitized 503.
            assert exc_info.value.status_code == 503
            # Exactly ONE audit event (inner emitted, outer suppressed).
            assert mock_logger.info.call_count == 1

    @pytest.mark.asyncio
    async def test_hydration_failure_emits_one_audit(self, fresh_manager):
        """Durable hydration failure emits exactly one audit event."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=None,  # Missing DualCache.
            ),
            patch(f"{_CLEAR_MOD}.logger") as mock_logger,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            assert exc_info.value.status_code == 503
            assert exc_info.value.detail["error"] == "cache_unavailable"
            assert mock_logger.info.call_count == 1
            payload = mock_logger.info.call_args[0][1]
            assert payload["event"] == "aawm_cooldown_clear_failure"
            assert payload["error_code"] == "cache_unavailable"

    @pytest.mark.asyncio
    async def test_drain_failure_emits_one_audit(self, fresh_manager):
        """Publication drain timeout emits exactly one conflict audit event."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        _seed_lane_keys(fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[cd_key])

        # Create a stuck publication intent that never completes.
        fresh_manager.publication_intents.create(
            alias_family="codex",
            cooldown_keys=frozenset({cd_key}),
            identity_hash=id_hash,
        )

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({cd_key}), cardinality=1
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=active_inspection,
            ),
            patch(f"{_CLEAR_MOD}.logger") as mock_logger,
            patch(
                f"{_CLEAR_MOD}._PUBLICATION_DRAIN_TIMEOUT_SECONDS",
                0.01,  # Very short timeout for test speed.
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            assert exc_info.value.status_code == 409
            assert exc_info.value.detail["error"] == "publication_drain_timeout"
            assert mock_logger.info.call_count == 1
            payload = mock_logger.info.call_args[0][1]
            assert payload["event"] == "aawm_cooldown_clear_conflict"
            assert payload["error_code"] == "publication_drain_timeout"

    @pytest.mark.asyncio
    async def test_inspection_failure_emits_one_audit(self, fresh_manager):
        """Prior-state inspection failure emits exactly one audit event."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        _seed_lane_keys(fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[cd_key])

        # Hydration succeeds, but prior-state inspection fails.
        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({cd_key}), cardinality=1
        )
        call_count = 0

        async def failing_inspect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: hydration (succeeds).
                return active_inspection
            # Second call: prior-state inspection (fails).
            raise RuntimeError("redis connection lost")

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                side_effect=failing_inspect,
            ),
            patch(f"{_CLEAR_MOD}.logger") as mock_logger,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            assert exc_info.value.status_code == 503
            assert mock_logger.info.call_count == 1
            payload = mock_logger.info.call_args[0][1]
            assert payload["event"] == "aawm_cooldown_clear_failure"
            assert payload["error_code"] == "inspection_failed"

    @pytest.mark.asyncio
    async def test_execution_failure_emits_one_audit(self, fresh_manager):
        """Clear execution failure emits exactly one audit event."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        _seed_lane_keys(fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[cd_key])

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({cd_key}), cardinality=1
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=active_inspection,
            ),
            patch(
                f"{_CLEAR_MOD}.clear_cooldown_transaction",
                new_callable=AsyncMock,
                side_effect=RuntimeError("redis down"),
            ),
            patch(f"{_CLEAR_MOD}.logger") as mock_logger,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            assert exc_info.value.status_code == 503
            assert mock_logger.info.call_count == 1
            payload = mock_logger.info.call_args[0][1]
            assert payload["event"] == "aawm_cooldown_clear_failure"
            assert payload["error_code"] == "clear_failed"

    @pytest.mark.asyncio
    async def test_no_duplicate_audit_on_post_resolution_failure(self, fresh_manager):
        """Post-resolution failures emit exactly one audit event (inner catch),
        not two (inner + outer)."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=None,  # Fail at hydration.
            ),
            patch(f"{_CLEAR_MOD}.logger") as mock_logger,
        ):
            with pytest.raises(HTTPException):
                await handle_cooldown_clear(request, admin)
            # Exactly ONE audit event, not two.
            assert mock_logger.info.call_count == 1

    @pytest.mark.asyncio
    async def test_audit_event_uses_partial_target_for_pre_resolution(self, fresh_manager):
        """Finding 5/6: pre-resolution failures use fixed safe labels, never raw caller strings."""
        request = _make_request({"alias": "test", "ingress": "codex", "bad": "x"})
        admin = _make_real_admin_key_dict()

        with (
            patch(f"{_CLEAR_MOD}.logger") as mock_logger,
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
        ):
            with pytest.raises(HTTPException):
                await handle_cooldown_clear(request, admin)
            payload = mock_logger.info.call_args[0][1]
            # Finding 5: fixed safe label, never raw caller alias.
            assert payload["target_description"] == "alias_present"
            assert payload["ingress"] == "codex"
            # No raw caller strings or internal state leaked.
            payload_str = str(payload)
            assert "traceback" not in payload_str.lower()
            assert "exception" not in payload_str.lower()
            assert "test" not in payload["target_description"]


# ===========================================================================
# Finding 1: already-leading unindexed publication drain tests
# ===========================================================================


class TestFinding1UnindexedPublicationDrain:
    """Finding 1: drain by identity catches unindexed leaders."""

    @pytest.fixture()
    def fresh_manager(self):
        return AliasRoutingStateManager()

    @pytest.mark.asyncio
    async def test_drain_awaits_unindexed_leader_by_identity(self, fresh_manager):
        """Already-LEADER intent with no cooldown_keys is drained by identity scan."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _drain_publication_intents,
        )

        registry = fresh_manager.publication_intents
        # Create a leader intent with identity but NO cooldown_keys registered.
        intent = registry.create(
            alias_family="codex",
            cooldown_keys=frozenset(),
            identity_hash="ident-abc",
        )
        assert not intent.done.is_set()

        # Drain should find it by identity and await it.
        async def complete_soon():
            await asyncio.sleep(0.05)
            intent.complete()

        task = asyncio.create_task(complete_soon())
        # Should not raise (completes within timeout).
        await _drain_publication_intents(
            registry, "codex", [], identity_hashes=frozenset({"ident-abc"})
        )
        await task
        assert intent.done.is_set()

    @pytest.mark.asyncio
    async def test_drain_rescan_after_completion(self, fresh_manager):
        """After first intent completes, rescan finds a second intent."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _drain_publication_intents,
        )

        registry = fresh_manager.publication_intents
        intent1 = registry.create(
            alias_family="codex",
            cooldown_keys=frozenset(),
            identity_hash="ident-x",
        )
        intent2 = registry.create(
            alias_family="codex",
            cooldown_keys=frozenset(),
            identity_hash="ident-x",
        )

        async def complete_staggered():
            await asyncio.sleep(0.02)
            intent1.complete()
            await asyncio.sleep(0.02)
            intent2.complete()

        task = asyncio.create_task(complete_staggered())
        await _drain_publication_intents(
            registry, "codex", [], identity_hashes=frozenset({"ident-x"})
        )
        await task
        assert intent1.done.is_set()
        assert intent2.done.is_set()

    @pytest.mark.asyncio
    async def test_drain_timeout_raises_409(self, fresh_manager):
        """Bounded timeout raises 409 conflict when intent never completes."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _drain_publication_intents,
        )
        import litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear as cc_mod

        registry = fresh_manager.publication_intents
        # Intent that never completes.
        registry.create(
            alias_family="codex",
            cooldown_keys=frozenset(),
            identity_hash="ident-stuck",
        )

        # Patch timeout to be very short for test speed.
        original_timeout = cc_mod._PUBLICATION_DRAIN_TIMEOUT_SECONDS
        cc_mod._PUBLICATION_DRAIN_TIMEOUT_SECONDS = 0.1
        try:
            with pytest.raises(HTTPException) as exc_info:
                await _drain_publication_intents(
                    registry, "codex", [], identity_hashes=frozenset({"ident-stuck"})
                )
            assert exc_info.value.status_code == 409
            assert exc_info.value.detail["error"] == "publication_drain_timeout"
        finally:
            cc_mod._PUBLICATION_DRAIN_TIMEOUT_SECONDS = original_timeout

    @pytest.mark.asyncio
    async def test_drain_no_deadlock_with_key_and_identity(self, fresh_manager):
        """Drain by both keys and identities does not deadlock."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _drain_publication_intents,
        )

        registry = fresh_manager.publication_intents
        # Intent registered by key AND identity.
        intent = registry.create(
            alias_family="codex",
            cooldown_keys=frozenset({"key-1"}),
            identity_hash="ident-dl",
        )

        async def complete_soon():
            await asyncio.sleep(0.02)
            intent.complete()

        task = asyncio.create_task(complete_soon())
        # Should not deadlock: same intent found by both paths, deduplicated.
        await asyncio.wait_for(
            _drain_publication_intents(
                registry, "codex", ["key-1"], identity_hashes=frozenset({"ident-dl"})
            ),
            timeout=5.0,
        )
        await task

    @pytest.mark.asyncio
    async def test_new_claims_blocked_by_clear_during_drain(self, fresh_manager):
        """New claims remain BLOCKED_BY_CLEAR while reservation is active."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
            ClaimOutcome,
        )

        registry = fresh_manager.publication_intents
        # Install a clear reservation.
        registry.create_clear_reservation(
            alias_family="codex",
            identity_hashes=frozenset({"ident-blk"}),
            cooldown_keys=frozenset(),
        )
        # New claim should be blocked.
        result = registry.claim_publication_or_wait(
            alias_family="codex",
            cooldown_keys=frozenset({"key-new"}),
            identity_hash="ident-blk",
        )
        assert result.outcome is ClaimOutcome.BLOCKED_BY_CLEAR


# ===========================================================================
# Finding 2: OpenRouter targeted clear tests
# ===========================================================================


class TestFinding2OpenRouterTargetedClear:
    """Finding 2: targeted OpenRouter rate-limit/failure-circuit clear."""

    @pytest.fixture()
    def fresh_manager(self):
        return AliasRoutingStateManager()

    def _make_openrouter_target(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _ResolvedIdentity,
            _ResolvedTarget,
        )
        ident = _ResolvedIdentity(
            identity_hash="or-hash-1",
            provider="openrouter",
            model="openrouter/free",
            route_family="codex_openrouter_responses",
            lane_keys=["lane-or-1"],
        )
        return _ResolvedTarget(
            family="codex",
            identities=[ident],
            target_description="alias:test-or",
            ingress="codex",
        )

    @pytest.mark.asyncio
    async def test_targeted_clear_removes_matching_keys_only(self, fresh_manager):
        """Only matching OpenRouter keys are cleared; unrelated keys preserved."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _clear_openrouter_local_state,
        )
        import time as _time

        mgr = fresh_manager
        # Set matching key and unrelated key.
        now = _time.monotonic()
        mgr.openrouter_rate_limit.until_monotonic_by_key["openrouter/free"] = now + 100
        mgr.openrouter_rate_limit.until_monotonic_by_key["unrelated-model"] = now + 100
        mgr.openrouter_failure_circuit.until_monotonic_by_key["openrouter/free"] = now + 100
        mgr.openrouter_failure_circuit.until_monotonic_by_key["other-model"] = now + 100

        target = self._make_openrouter_target()
        result = await _clear_openrouter_local_state(target, mgr)

        assert result["openrouter_keys_cleared"] > 0
        # Unrelated keys preserved.
        assert "unrelated-model" in mgr.openrouter_rate_limit.until_monotonic_by_key
        assert "other-model" in mgr.openrouter_failure_circuit.until_monotonic_by_key
        # Matching keys removed.
        assert "openrouter/free" not in mgr.openrouter_rate_limit.until_monotonic_by_key
        assert "openrouter/free" not in mgr.openrouter_failure_circuit.until_monotonic_by_key

    @pytest.mark.asyncio
    async def test_no_global_flush(self, fresh_manager):
        """Never performs a global flush of OpenRouter maps."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _clear_openrouter_local_state,
        )
        import time as _time

        mgr = fresh_manager
        now = _time.monotonic()
        for i in range(10):
            mgr.openrouter_rate_limit.until_monotonic_by_key[f"model-{i}"] = now + 100

        target = self._make_openrouter_target()
        await _clear_openrouter_local_state(target, mgr)

        # All unrelated keys still present.
        for i in range(10):
            if f"model-{i}" != "openrouter/free":
                assert f"model-{i}" in mgr.openrouter_rate_limit.until_monotonic_by_key

    @pytest.mark.asyncio
    async def test_concurrent_lock_safety(self, fresh_manager):
        """Concurrent clears under the lock do not corrupt state."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _clear_openrouter_local_state,
        )
        import time as _time

        mgr = fresh_manager
        now = _time.monotonic()
        mgr.openrouter_rate_limit.until_monotonic_by_key["openrouter/free"] = now + 100

        target = self._make_openrouter_target()
        # Run two concurrent clears.
        results = await asyncio.gather(
            _clear_openrouter_local_state(target, mgr),
            _clear_openrouter_local_state(target, mgr),
        )
        # Total cleared across both should be exactly 1 (idempotent).
        total = sum(r["openrouter_keys_cleared"] for r in results)
        assert total == 1

    def test_derive_keys_never_uses_identity_hash(self):
        """Derived keys never include identity_hash values."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _derive_openrouter_rate_limit_keys,
        )

        target = self._make_openrouter_target()
        keys = _derive_openrouter_rate_limit_keys(target)
        for key in keys:
            assert key != "or-hash-1"
            assert "identity" not in key.lower()

    def test_non_openrouter_candidates_produce_no_keys(self):
        """Non-OpenRouter candidates produce empty key list."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _ResolvedIdentity,
            _ResolvedTarget,
            _derive_openrouter_rate_limit_keys,
        )

        ident = _ResolvedIdentity(
            identity_hash="hash-1",
            provider="openai",
            model="gpt-4o",
            route_family="codex_openai_responses",
            lane_keys=["lane-1"],
        )
        target = _ResolvedTarget(
            family="codex",
            identities=[ident],
            target_description="alias:test",
            ingress="codex",
        )
        assert _derive_openrouter_rate_limit_keys(target) == []


# ===========================================================================
# Finding 3: all-or-none multi-identity clear + rollback tests
# ===========================================================================


class TestFinding3AllOrNoneRollback:
    """Finding 3: multi-identity all-or-none with rollback."""

    @pytest.fixture()
    def fresh_manager(self):
        return AliasRoutingStateManager()

    @pytest.mark.asyncio
    async def test_second_identity_failure_triggers_rollback(self, fresh_manager):
        """On second identity durable failure, first identity is rolled back."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _execute_clear,
            _ResolvedIdentity,
            _ResolvedTarget,
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
            ClearTransactionJournal,
            ClearTransactionResult,
        )

        mgr = fresh_manager
        ident1 = _ResolvedIdentity(
            identity_hash="h1", provider="openai", model="gpt-4o",
            route_family="codex_openai_responses", lane_keys=["k1"],
        )
        ident2 = _ResolvedIdentity(
            identity_hash="h2", provider="openai", model="gpt-4o-mini",
            route_family="codex_openai_responses", lane_keys=["k2"],
        )
        target = _ResolvedTarget(
            family="codex", identities=[ident1, ident2],
            target_description="alias:multi", ingress="codex",
        )

        journal1 = ClearTransactionJournal(
            transaction_id="txn-1", phase="committed", alias_family="codex",
            identity_hash="h1", cooldown_keys=["k1"], lane_members=["k1"],
            expected_members=["k1"], identity_key="id:k1",
            receipt_key="rcpt:1", receipt_ttl=300,
        )
        result1 = ClearTransactionResult(
            transaction_id="txn-1", phase="committed", journal=journal1,
            keys_deleted=1, members_removed=1,
        )

        call_count = [0]

        async def mock_durable_clear(*, family, identity_hash, cooldown_keys, lane_members):
            call_count[0] += 1
            if identity_hash == "h1":
                return result1
            # Second identity fails.
            raise HTTPException(status_code=503, detail={"error": "clear_failed", "message": "fail"})

        rollback_called = []

        async def mock_rollback(*, family, committed_results):
            rollback_called.extend(committed_results)

        with (
            patch(f"{_CLEAR_MOD}._execute_durable_clear", side_effect=mock_durable_clear),
            patch(f"{_CLEAR_MOD}._rollback_committed_results", side_effect=mock_rollback),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _execute_clear(target, mgr)
            assert exc_info.value.status_code == 503

        # Rollback was called with the first result.
        assert len(rollback_called) == 1
        assert rollback_called[0].transaction_id == "txn-1"

    @pytest.mark.asyncio
    async def test_rollback_success_no_error(self, fresh_manager):
        """Successful rollback after durable failure does not add extra error."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _rollback_committed_results,
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
            ClearTransactionJournal,
            ClearTransactionResult,
        )

        journal = ClearTransactionJournal(
            transaction_id="txn-ok", phase="committed", alias_family="codex",
            identity_hash="h1", cooldown_keys=["k1"], lane_members=["k1"],
            expected_members=["k1"], identity_key="id:k1",
            receipt_key="rcpt:ok", receipt_ttl=300,
        )
        result = ClearTransactionResult(
            transaction_id="txn-ok", phase="committed", journal=journal,
            keys_deleted=1, members_removed=1,
        )

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable.rollback_clear_transaction",
            new_callable=AsyncMock,
        ) as mock_rb:
            mock_rb.return_value = None
            # Should not raise.
            await _rollback_committed_results(family="codex", committed_results=[result])
            mock_rb.assert_called_once()

    @pytest.mark.asyncio
    async def test_rollback_drift_raises_sanitized(self, fresh_manager):
        """Rollback drift raises sanitized rollback_failure."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _rollback_committed_results,
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
            ClearTransactionJournal,
            ClearTransactionResult,
            RollbackDriftError,
        )

        journal = ClearTransactionJournal(
            transaction_id="txn-drift", phase="committed", alias_family="codex",
            identity_hash="h1", cooldown_keys=["k1"], lane_members=["k1"],
            expected_members=["k1"], identity_key="id:k1",
            receipt_key="rcpt:drift", receipt_ttl=300,
        )
        result = ClearTransactionResult(
            transaction_id="txn-drift", phase="committed", journal=journal,
            keys_deleted=1, members_removed=1,
        )

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable.rollback_clear_transaction",
            new_callable=AsyncMock,
            side_effect=RollbackDriftError(
                phase="committed", family="codex",
                transaction_id_prefix="txn-drift", identity_prefix="h1",
                key_count=1, exception_classes=(),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _rollback_committed_results(family="codex", committed_results=[result])
            assert exc_info.value.status_code == 503
            assert exc_info.value.detail["error"] == "rollback_failure"

    @pytest.mark.asyncio
    async def test_rollback_receipt_missing_raises_sanitized(self, fresh_manager):
        """Rollback receipt missing raises sanitized rollback_failure."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _rollback_committed_results,
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
            ClearTransactionJournal,
            ClearTransactionResult,
            RollbackReceiptMissingError,
        )

        journal = ClearTransactionJournal(
            transaction_id="txn-miss", phase="committed", alias_family="codex",
            identity_hash="h1", cooldown_keys=["k1"], lane_members=["k1"],
            expected_members=["k1"], identity_key="id:k1",
            receipt_key="rcpt:miss", receipt_ttl=300,
        )
        result = ClearTransactionResult(
            transaction_id="txn-miss", phase="committed", journal=journal,
            keys_deleted=1, members_removed=1,
        )

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable.rollback_clear_transaction",
            new_callable=AsyncMock,
            side_effect=RollbackReceiptMissingError(
                phase="committed", family="codex",
                transaction_id_prefix="txn-miss", identity_prefix="h1",
                key_count=1, exception_classes=(),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _rollback_committed_results(family="codex", committed_results=[result])
            assert exc_info.value.status_code == 503
            assert exc_info.value.detail["error"] == "rollback_failure"

    @pytest.mark.asyncio
    async def test_local_failure_after_durable_triggers_rollback(  # noqa: PLR0915
        self, fresh_manager
    ):
        """Local/postcondition failure after durable commits triggers rollback.

        Seeds EVERY targeted local state the clear path mutates (family
        positive/negative cooldown maps, evidence events, per-key generation,
        alias-scoped Codex failure evidence, lane-identity index membership,
        and targeted OpenRouter rate-limit/failure-circuit entries) plus
        unrelated state and session affinity.  On postcondition failure the
        durable receipts are rolled back AND every captured local preimage is
        restored exactly, while unrelated state and affinity are preserved.
        """
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _execute_clear,
            _ResolvedIdentity,
            _ResolvedTarget,
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.classification import (
            _KeyCooldownState,
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
            ClearTransactionJournal,
            ClearTransactionResult,
        )

        mgr = fresh_manager
        ident = _ResolvedIdentity(
            identity_hash="h1", provider="openai", model="gpt-4o",
            route_family="codex_openai_responses", lane_keys=["k1"],
        )
        target = _ResolvedTarget(
            family="codex",
            canonical_aliases=("local-fail",),
            identities=[ident],
            target_description="alias:local-fail", ingress="codex",
        )

        # --- Seed targeted local state (preimage values) -------------------
        now = time.monotonic()
        fam = mgr.codex
        fam.cooldown_until_monotonic_by_key["k1"] = now + 100.0
        fam.cooldown_negative_until_monotonic_by_key["k1"] = now + 50.0
        fam.evidence_events_by_key["k1"] = [now - 5.0, now - 1.0]
        fam.cooldown_generation_by_key["k1"] = 3
        # Alias-scoped Codex failure-evidence key + marker state.
        gate_ks = _KeyCooldownState(
            attempt=2, cooled_until_monotonic=now + 75.0,
            probe_in_flight=True, last_scope="scope-a", last_class_name="Cls",
        )
        failure_gate = mgr.codex_failure_evidence_gate.gate_for_alias(
            canonical_alias="local-fail",
            create=True,
        )
        assert failure_gate is not None
        failure_gate._key_state["k1"] = gate_ks
        failure_gate._family_state.evidence_events_by_key["k1"] = [now - 2.0]
        # Lane-identity index membership: targeted lane plus an unrelated
        # lane under the SAME identity (must survive restoration).
        mgr.lane_identity_index.register(identity_hash="h1", lane_key="k1")
        mgr.lane_identity_index.register(identity_hash="h1", lane_key="k-unrelated-lane")
        # Targeted OpenRouter rate-limit + failure-circuit entries.
        mgr.openrouter_rate_limit.until_monotonic_by_key["or-model-x"] = now + 30.0
        mgr.openrouter_failure_circuit.until_monotonic_by_key["or-model-x"] = now + 40.0

        # --- Seed unrelated state that must be preserved exactly -----------
        fam.cooldown_until_monotonic_by_key["unrelated-key"] = now + 999.0
        fam.session_affinity_by_key["session-1"] = {
            "provider": "openai", "model": "gpt-4o",
            "route_family": "codex_openai_responses", "last_resort": False,
            "expires_at_monotonic": now + 500.0, "affinity_state_source": "memory",
        }
        mgr.lane_identity_index.register(identity_hash="h-other", lane_key="k-other")
        mgr.openrouter_rate_limit.until_monotonic_by_key["or-unrelated"] = now + 88.0

        journal = ClearTransactionJournal(
            transaction_id="txn-lf", phase="committed", alias_family="codex",
            identity_hash="h1", cooldown_keys=["k1"], lane_members=["k1"],
            expected_members=["k1"], identity_key="id:k1",
            receipt_key="rcpt:lf", receipt_ttl=300,
        )
        result = ClearTransactionResult(
            transaction_id="txn-lf", phase="committed", journal=journal,
            keys_deleted=1, members_removed=1,
        )

        async def mock_durable_clear(*, family, identity_hash, cooldown_keys, lane_members):
            return result

        rollback_called = []

        async def mock_rollback(*, family, committed_results):
            rollback_called.extend(committed_results)

        with (
            patch(f"{_CLEAR_MOD}._execute_durable_clear", side_effect=mock_durable_clear),
            patch(f"{_CLEAR_MOD}._verify_postconditions", side_effect=HTTPException(
                status_code=500, detail={"error": "postcondition_failure", "message": "fail"}
            )),
            patch(f"{_CLEAR_MOD}._rollback_committed_results", side_effect=mock_rollback),
            patch(
                f"{_CLEAR_MOD}._derive_openrouter_rate_limit_keys",
                return_value=["or-model-x"],
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _execute_clear(target, mgr)
            assert exc_info.value.status_code == 500

        assert len(rollback_called) == 1

        # --- Exact restoration of every targeted local preimage ------------
        assert fam.cooldown_until_monotonic_by_key["k1"] == now + 100.0
        assert fam.cooldown_negative_until_monotonic_by_key["k1"] == now + 50.0
        assert fam.evidence_events_by_key["k1"] == [now - 5.0, now - 1.0]
        assert fam.cooldown_generation_by_key["k1"] == 3
        restored_gate = mgr.codex_failure_evidence_gate.gate_for_alias(
            canonical_alias="local-fail"
        )
        assert restored_gate is not None
        restored_ks = restored_gate._key_state["k1"]
        assert restored_ks == gate_ks
        assert restored_ks is not gate_ks  # independent restored copy
        assert restored_gate._family_state.evidence_events_by_key["k1"] == [now - 2.0]
        assert mgr.lane_identity_index.lanes_for("h1") == frozenset({"k1", "k-unrelated-lane"})
        assert mgr.openrouter_rate_limit.until_monotonic_by_key["or-model-x"] == now + 30.0
        assert mgr.openrouter_failure_circuit.until_monotonic_by_key["or-model-x"] == now + 40.0

        # --- Unrelated state + session affinity preserved exactly ----------
        assert fam.cooldown_until_monotonic_by_key["unrelated-key"] == now + 999.0
        assert fam.session_affinity_by_key["session-1"]["model"] == "gpt-4o"
        assert mgr.lane_identity_index.lanes_for("h-other") == frozenset({"k-other"})
        assert mgr.openrouter_rate_limit.until_monotonic_by_key["or-unrelated"] == now + 88.0

    @pytest.mark.asyncio
    async def test_local_restoration_failure_fails_closed(self, fresh_manager):
        """Unprovable local restoration after rollback fails closed (503).

        When durable receipts roll back but the process-local preimage
        restoration cannot be proven, the endpoint must fail closed with
        sanitized indeterminate semantics rather than return the original
        postcondition failure.
        """
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _execute_clear,
            _ResolvedIdentity,
            _ResolvedTarget,
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
            ClearTransactionJournal,
            ClearTransactionResult,
        )

        mgr = fresh_manager
        ident = _ResolvedIdentity(
            identity_hash="h1", provider="openai", model="gpt-4o",
            route_family="codex_openai_responses", lane_keys=["k1"],
        )
        target = _ResolvedTarget(
            family="codex", identities=[ident],
            target_description="alias:restore-fail", ingress="codex",
        )
        mgr.codex.cooldown_until_monotonic_by_key["k1"] = time.monotonic() + 100.0

        journal = ClearTransactionJournal(
            transaction_id="txn-rf", phase="committed", alias_family="codex",
            identity_hash="h1", cooldown_keys=["k1"], lane_members=["k1"],
            expected_members=["k1"], identity_key="id:k1",
            receipt_key="rcpt:rf", receipt_ttl=300,
        )
        result = ClearTransactionResult(
            transaction_id="txn-rf", phase="committed", journal=journal,
            keys_deleted=1, members_removed=1,
        )

        async def mock_durable_clear(*, family, identity_hash, cooldown_keys, lane_members):
            return result

        rollback_called = []

        async def mock_rollback(*, family, committed_results):
            rollback_called.extend(committed_results)

        with (
            patch(f"{_CLEAR_MOD}._execute_durable_clear", side_effect=mock_durable_clear),
            patch(f"{_CLEAR_MOD}._verify_postconditions", side_effect=HTTPException(
                status_code=500, detail={"error": "postcondition_failure", "message": "fail"}
            )),
            patch(f"{_CLEAR_MOD}._rollback_committed_results", side_effect=mock_rollback),
            patch(f"{_CLEAR_MOD}._restore_local_preimage", return_value=False),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _execute_clear(target, mgr)
            assert exc_info.value.status_code == 503
            assert exc_info.value.detail["error"] == "indeterminate_clear"

        # Durable rollback still attempted before fail-closed.
        assert len(rollback_called) == 1

    @pytest.mark.asyncio
    async def test_local_restoration_runs_even_when_durable_rollback_fails(
        self, fresh_manager
    ):
        """Local preimages are restored even when durable rollback raises.

        Regression: on postcondition failure, durable rollback raising a
        sanitized rollback_failure must NOT skip local restoration.  Local
        preimages are restored first, and because restoration is proven, the
        durable rollback error propagates (not the original postcondition
        error, and not indeterminate_clear).
        """
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _execute_clear,
            _ResolvedIdentity,
            _ResolvedTarget,
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
            ClearTransactionJournal,
            ClearTransactionResult,
        )

        mgr = fresh_manager
        ident = _ResolvedIdentity(
            identity_hash="h1", provider="openai", model="gpt-4o",
            route_family="codex_openai_responses", lane_keys=["k1"],
        )
        target = _ResolvedTarget(
            family="codex", identities=[ident],
            target_description="alias:rb-fail", ingress="codex",
        )

        now = time.monotonic()
        fam = mgr.codex
        fam.cooldown_until_monotonic_by_key["k1"] = now + 100.0
        fam.cooldown_negative_until_monotonic_by_key["k1"] = now + 50.0
        fam.evidence_events_by_key["k1"] = [now - 1.0]
        fam.cooldown_generation_by_key["k1"] = 2
        mgr.lane_identity_index.register(identity_hash="h1", lane_key="k1")
        mgr.openrouter_rate_limit.until_monotonic_by_key["or-model-x"] = now + 30.0

        journal = ClearTransactionJournal(
            transaction_id="txn-rbf", phase="committed", alias_family="codex",
            identity_hash="h1", cooldown_keys=["k1"], lane_members=["k1"],
            expected_members=["k1"], identity_key="id:k1",
            receipt_key="rcpt:rbf", receipt_ttl=300,
        )
        result = ClearTransactionResult(
            transaction_id="txn-rbf", phase="committed", journal=journal,
            keys_deleted=1, members_removed=1,
        )

        async def mock_durable_clear(*, family, identity_hash, cooldown_keys, lane_members):
            return result

        async def mock_rollback_fail(*, family, committed_results):
            raise HTTPException(
                status_code=503,
                detail={"error": "rollback_failure", "message": "drift"},
            )

        with (
            patch(f"{_CLEAR_MOD}._execute_durable_clear", side_effect=mock_durable_clear),
            patch(f"{_CLEAR_MOD}._verify_postconditions", side_effect=HTTPException(
                status_code=500, detail={"error": "postcondition_failure", "message": "fail"}
            )),
            patch(f"{_CLEAR_MOD}._rollback_committed_results", side_effect=mock_rollback_fail),
            patch(
                f"{_CLEAR_MOD}._derive_openrouter_rate_limit_keys",
                return_value=["or-model-x"],
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _execute_clear(target, mgr)
            # Durable rollback error propagates (restoration was proven).
            assert exc_info.value.status_code == 503
            assert exc_info.value.detail["error"] == "rollback_failure"

        # Local preimages were restored despite durable rollback failure.
        assert fam.cooldown_until_monotonic_by_key["k1"] == now + 100.0
        assert fam.cooldown_negative_until_monotonic_by_key["k1"] == now + 50.0
        assert fam.evidence_events_by_key["k1"] == [now - 1.0]
        assert fam.cooldown_generation_by_key["k1"] == 2
        assert mgr.lane_identity_index.lanes_for("h1") == frozenset({"k1"})
        assert mgr.openrouter_rate_limit.until_monotonic_by_key["or-model-x"] == now + 30.0

    @pytest.mark.asyncio
    async def test_unrelated_state_preserved_after_clear(self, fresh_manager):
        """Unrelated cooldown keys are preserved after a targeted clear."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _execute_clear,
            _ResolvedIdentity,
            _ResolvedTarget,
        )
        import time as _time

        mgr = fresh_manager
        now = _time.monotonic()
        # Set unrelated cooldown.
        mgr.codex.cooldown_until_monotonic_by_key["unrelated-key"] = now + 999

        ident = _ResolvedIdentity(
            identity_hash="h1", provider="openai", model="gpt-4o",
            route_family="codex_openai_responses", lane_keys=["k1"],
        )
        target = _ResolvedTarget(
            family="codex", identities=[ident],
            target_description="alias:preserve", ingress="codex",
        )

        with (
            patch(f"{_CLEAR_MOD}._execute_durable_clear", return_value=None),
            patch(f"{_CLEAR_MOD}._verify_postconditions", new_callable=AsyncMock),
        ):
            await _execute_clear(target, mgr)

        # Unrelated key preserved.
        assert "unrelated-key" in mgr.codex.cooldown_until_monotonic_by_key


# ===========================================================================
# Finding 4: route-auth audit ASGI tests
# ===========================================================================


class TestFinding4RouteAuthAudit:
    """Finding 4: route-specific auth dependency emits audit on failure."""

    @pytest.mark.asyncio
    async def test_absent_token_emits_one_audit(self):
        """Absent token triggers exactly one sanitized audit event."""
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _cfg004_cooldown_clear_auth_dependency,
        )

        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.method = "POST"
        mock_request.url = MagicMock()
        mock_request.url.path = "/aawm/alias-routing/cooldowns/clear"

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth",
                new_callable=AsyncMock,
                side_effect=HTTPException(status_code=401, detail="Unauthorized"),
            ),
            patch("logging.getLogger") as mock_get_logger,
        ):
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            with pytest.raises(HTTPException) as exc_info:
                await _cfg004_cooldown_clear_auth_dependency(
                    request=mock_request, api_key=""
                )
            assert exc_info.value.status_code == 401
            # Exactly one audit event.
            assert mock_logger.info.call_count == 1
            payload = mock_logger.info.call_args[0][1]
            assert payload["event"] == "aawm_cooldown_clear_auth_failure"
            assert payload["target_description"] == "target_unavailable"
            # No secrets.
            payload_str = str(payload)
            assert "Bearer" not in payload_str
            assert "token" not in payload_str.lower() or "token" in "target_unavailable"

    @pytest.mark.asyncio
    async def test_malformed_token_emits_one_audit(self):
        """Malformed token triggers exactly one sanitized audit event."""
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _cfg004_cooldown_clear_auth_dependency,
        )

        mock_request = MagicMock()
        mock_request.headers = {"authorization": "Bearer garbage!!!"}
        mock_request.method = "POST"

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth",
                new_callable=AsyncMock,
                side_effect=HTTPException(status_code=401, detail="Invalid key"),
            ),
            patch("logging.getLogger") as mock_get_logger,
        ):
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            with pytest.raises(HTTPException):
                await _cfg004_cooldown_clear_auth_dependency(
                    request=mock_request, api_key="Bearer garbage!!!"
                )
            assert mock_logger.info.call_count == 1

    @pytest.mark.asyncio
    async def test_successful_auth_returns_user_api_key_auth(self):
        """Successful auth returns UserAPIKeyAuth without audit."""
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _cfg004_cooldown_clear_auth_dependency,
        )
        from litellm.proxy._types import UserAPIKeyAuth

        mock_request = MagicMock()
        expected = UserAPIKeyAuth()

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth",
                new_callable=AsyncMock,
                return_value=expected,
            ),
            patch("logging.getLogger") as mock_get_logger,
        ):
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            result = await _cfg004_cooldown_clear_auth_dependency(
                request=mock_request, api_key="sk-valid"
            )
            assert result is expected
            # No audit on success.
            mock_logger.info.assert_not_called()


# ===========================================================================
# Finding 5: strict schema + secret-shaped tests
# ===========================================================================


class TestFinding5StrictSchema:
    """Finding 5: explicit null/empty/whitespace supplied fields are invalid."""

    def test_explicit_null_alias_rejected(self):
        """Explicit null alias is invalid even if unused."""
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_request(
                {"alias": None, "provider": "openai", "model": "gpt-4o", "ingress": "codex"}
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "invalid_field_value"

    def test_empty_string_alias_rejected(self):
        """Empty string alias is invalid."""
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_request(
                {"alias": "", "ingress": "codex"}
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "invalid_field_value"

    def test_whitespace_only_model_rejected(self):
        """Whitespace-only model is invalid when supplied."""
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_request(
                {"provider": "openai", "model": "   ", "ingress": "codex"}
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "invalid_field_value"

    def test_omitted_fields_allowed(self):
        """Omitted fields are allowed per alias XOR exact form."""
        # Alias form: provider/model omitted is fine.
        req = _parse_and_validate_request({"alias": "my-alias", "ingress": "codex"})
        assert req.alias == "my-alias"
        assert req.provider is None
        assert req.model is None

    def test_secret_shaped_alias_passes_validation_no_echo(self):
        """Secret-shaped alias passes validation (valid string) but is never echoed in errors."""
        # A secret-shaped alias is a valid non-empty string; validation passes.
        req = _parse_and_validate_request(
            {"alias": "sk-ant-api03-secret-value-here", "ingress": "codex"}
        )
        assert req.alias == "sk-ant-api03-secret-value-here"
        # The real protection: error messages from later stages never echo it.
        # Forbidden field names ARE rejected without echo:
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_request(
                {"alias": "sk-ant-api03-secret", "ingress": "codex", "raw_key": "x"}
            )
        assert exc_info.value.status_code == 400
        assert "sk-ant-api03-secret" not in str(exc_info.value.detail)

    def test_secret_shaped_forbidden_field_no_echo(self):
        """Forbidden field names are rejected without echoing values."""
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_request(
                {"alias": "test", "ingress": "codex", "identity_hash": "secret-hash-value"}
            )
        assert exc_info.value.status_code == 400
        detail_str = str(exc_info.value.detail)
        assert "secret-hash-value" not in detail_str

    def test_explicit_null_provider_with_alias_rejected(self):
        """Explicit null provider is invalid even when alias form is used."""
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_request(
                {"alias": "my-alias", "provider": None, "ingress": "codex"}
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "invalid_field_value"


# ===========================================================================
# Finding 6: response/audit completeness tests
# ===========================================================================


class TestFinding6ResponseAuditCompleteness:
    """Finding 6: failure responses include safe target classification."""

    @pytest.fixture()
    def fresh_manager(self):
        return AliasRoutingStateManager()

    @pytest.mark.asyncio
    async def test_failure_audit_includes_environment_and_namespace(self, fresh_manager):
        """Failure audit includes environment and namespace fields."""
        request = _make_request({"alias": "test", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        with (
            patch(f"{_CLEAR_MOD}.logger") as mock_logger,
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(f"{_CLEAR_MOD}._check_topology_gate", side_effect=HTTPException(
                status_code=503, detail={"error": "topology_gate_closed", "message": "closed"}
            )),
            patch(f"{_CLEAR_MOD}._check_admin_auth"),
        ):
            with pytest.raises(HTTPException):
                await handle_cooldown_clear(request, admin)
            payload = mock_logger.info.call_args[0][1]
            assert "environment" in payload
            assert "namespace" in payload
            assert "result" in payload
            assert payload["result"] == "error"

    @pytest.mark.asyncio
    async def test_exactly_one_event_per_attempt(self, fresh_manager):
        """Exactly one audit event per attempt regardless of failure point."""
        request = _make_request({"alias": "test", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        with (
            patch(f"{_CLEAR_MOD}.logger") as mock_logger,
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(f"{_CLEAR_MOD}._check_admin_auth", side_effect=HTTPException(
                status_code=403, detail={"error": "forbidden", "message": "no"}
            )),
        ):
            with pytest.raises(HTTPException):
                await handle_cooldown_clear(request, admin)
            assert mock_logger.info.call_count == 1

    @pytest.mark.asyncio
    async def test_pre_auth_event_uses_safe_unavailable_fields(self, fresh_manager):
        """Pre-auth/prevalidation events use safe unavailable fields."""
        request = _make_request({"ingress": "codex"})  # Missing target.
        admin = _make_real_admin_key_dict()

        with (
            patch(f"{_CLEAR_MOD}.logger") as mock_logger,
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(f"{_CLEAR_MOD}._check_admin_auth"),
        ):
            with pytest.raises(HTTPException):
                await handle_cooldown_clear(request, admin)
            payload = mock_logger.info.call_args[0][1]
            assert payload["target_description"] == "target_unavailable"


# ===========================================================================
# Defect 1: Post-drain rehydration tests
# ===========================================================================


class TestPostDrainRehydration:
    """Defect 1: after identity-based drain completes, rehydrate/rescan both
    local lane index and bounded durable identity membership for all target
    identities, union newly published lanes, extend the active reservation
    with them, and only then inspect/execute."""

    @pytest.mark.asyncio
    async def test_leader_publishes_on_completion_cleared_not_zero(self, fresh_manager):
        """Already-LEADER with no initial index publishes on completion:
        endpoint must clear that new lane and cannot return cleared with
        zero keys while it remains."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        published_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct-pub"

        # Simulate: initial hydration finds nothing (no index, no durable).
        # Post-drain rehydration finds the published key in durable.
        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )
        published_inspection = _make_identity_inspection(
            exists=True, members=frozenset({published_key}), cardinality=1
        )

        call_count = 0

        async def inspect_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            # Calls 1: initial hydration (absent)
            # Call 2: post-drain rehydration (published!)
            # Call 3: prior-state (active because published)
            # Call 4: durable-clear inspect (active)
            # Call 5: postcondition (absent)
            if call_count == 1:
                return absent_inspection
            if call_count in (2, 3, 4):
                return published_inspection
            return absent_inspection

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                side_effect=inspect_side_effect,
            ),
            patch(
                f"{_CLEAR_MOD}.clear_cooldown_transaction",
                new_callable=AsyncMock,
                return_value=_make_txn_result(),
            ),
            patch(
                f"{_CLEAR_MOD}.verify_aawm_alias_routing_durable_absence",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await handle_cooldown_clear(request, admin)
            # Must clear the published lane, NOT return cleared with zero keys.
            assert result["result"] == "cleared"
            assert result["keys_cleared"] >= 1

    @pytest.mark.asyncio
    async def test_post_drain_instability_fails_closed(self, fresh_manager):
        """If lane key set changes unexpectedly after drain (bounded stability
        check), endpoint fails closed with 409 post_drain_instability."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        key1 = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        key2 = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct2"

        # Seed initial lane so hydration finds key1.
        _seed_lane_keys(fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[key1])

        # After post-drain rehydration, inject key2 into the local index
        # to simulate instability.
        original_hydrate = _hydrate_from_local_index
        hydrate_call_count = 0

        def destabilizing_hydrate(target, mgr):
            nonlocal hydrate_call_count
            hydrate_call_count += 1
            original_hydrate(target, mgr)
            # On the stability-check hydration (3rd call), inject a new key.
            if hydrate_call_count == 3:
                mgr.lane_identity_index.register_batch(
                    identity_hash=id_hash, lane_keys=[key2]
                )
                # Re-run hydration to pick up the injected key.
                original_hydrate(target, mgr)

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({key1}), cardinality=1
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=active_inspection,
            ),
            patch(f"{_CLEAR_MOD}._hydrate_from_local_index", side_effect=destabilizing_hydrate),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            assert exc_info.value.status_code == 409
            assert exc_info.value.detail["error"] == "post_drain_instability"


# ===========================================================================
# Defect 2: OpenRouter-only clear (no lane keys) tests
# ===========================================================================


class TestOpenRouterOnlyClear:
    """Defect 2: do not early-return from _execute_clear solely because lane
    key set is empty when candidate-derived OpenRouter adapter/upstream blocker
    keys are active."""

    @pytest.mark.asyncio
    async def test_openrouter_only_clear_returns_cleared(self, fresh_manager):
        """OpenRouter rate-limit entries exist but no lane keys: endpoint
        clears OpenRouter state and returns cleared."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _execute_clear,
            _ResolvedIdentity,
            _ResolvedTarget,
        )

        mgr = fresh_manager
        ident = _ResolvedIdentity(
            identity_hash="h-or", provider="openrouter",
            model="anthropic/claude-3-opus", route_family="codex_openrouter_adapter",
            lane_keys=[],
        )
        target = _ResolvedTarget(
            family="codex", identities=[ident],
            target_description="alias:or-test", ingress="codex",
        )

        # Seed OpenRouter rate-limit state.
        now = time.monotonic()
        mgr.openrouter_rate_limit.until_monotonic_by_key["anthropic/claude-3-opus"] = now + 300
        mgr.openrouter_failure_circuit.until_monotonic_by_key["anthropic/claude-3-opus"] = now + 300
        # Seed unrelated OpenRouter state.
        mgr.openrouter_rate_limit.until_monotonic_by_key["unrelated-model"] = now + 999

        with (
            patch(
                f"{_CLEAR_MOD}._derive_openrouter_rate_limit_keys",
                return_value=["anthropic/claude-3-opus"],
            ),
        ):
            keys_cleared, members_removed = await _execute_clear(target, mgr)

        assert keys_cleared >= 1
        # Target keys cleared.
        assert "anthropic/claude-3-opus" not in mgr.openrouter_rate_limit.until_monotonic_by_key
        assert "anthropic/claude-3-opus" not in mgr.openrouter_failure_circuit.until_monotonic_by_key
        # Unrelated preserved.
        assert "unrelated-model" in mgr.openrouter_rate_limit.until_monotonic_by_key

    @pytest.mark.asyncio
    async def test_openrouter_no_state_returns_zero(self, fresh_manager):
        """No lane keys AND no active OpenRouter state: returns (0, 0)."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _execute_clear,
            _ResolvedIdentity,
            _ResolvedTarget,
        )

        mgr = fresh_manager
        ident = _ResolvedIdentity(
            identity_hash="h-or", provider="openrouter",
            model="anthropic/claude-3-opus", route_family="codex_openrouter_adapter",
            lane_keys=[],
        )
        target = _ResolvedTarget(
            family="codex", identities=[ident],
            target_description="alias:or-empty", ingress="codex",
        )

        with (
            patch(
                f"{_CLEAR_MOD}._derive_openrouter_rate_limit_keys",
                return_value=["anthropic/claude-3-opus"],
            ),
        ):
            keys_cleared, members_removed = await _execute_clear(target, mgr)

        assert keys_cleared == 0
        assert members_removed == 0

    @pytest.mark.asyncio
    async def test_no_lane_no_openrouter_returns_zero(self, fresh_manager):
        """No lane keys AND no OpenRouter candidates: returns (0, 0)."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _execute_clear,
            _ResolvedIdentity,
            _ResolvedTarget,
        )

        mgr = fresh_manager
        ident = _ResolvedIdentity(
            identity_hash="h-openai", provider="openai",
            model="gpt-4o", route_family="codex_openai_responses",
            lane_keys=[],
        )
        target = _ResolvedTarget(
            family="codex", identities=[ident],
            target_description="alias:empty", ingress="codex",
        )

        keys_cleared, members_removed = await _execute_clear(target, mgr)
        assert keys_cleared == 0
        assert members_removed == 0

    @pytest.mark.asyncio
    async def test_openrouter_only_full_endpoint_not_active(self, fresh_manager):
        """Full endpoint: OpenRouter candidate with no lane keys and no
        active OpenRouter state returns not_active."""
        snapshot = _make_snapshot(
            provider="openrouter",
            model="anthropic/claude-3-opus",
            route_family="codex_openrouter_adapter",
        )
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=absent_inspection,
            ),
            patch(
                f"{_CLEAR_MOD}._derive_openrouter_rate_limit_keys",
                return_value=["anthropic/claude-3-opus"],
            ),
            patch(
                f"{_CLEAR_MOD}._inspect_openrouter_prior_state",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            result = await handle_cooldown_clear(request, admin)
            assert result["result"] == "not_active"
            assert result["keys_cleared"] == 0


# ===========================================================================
# Defect 3: Post-inspection failure enrichment tests
# ===========================================================================


class TestPostInspectionFailureEnrichment:
    """Defect 3: once prior inspection has established source and bounded TTL,
    every later failure response and audit must include safe prior_state_source
    and bounded_remaining_ttl_seconds, plus resolved candidates/ingress/
    environment/namespace.  Pre-inspection failures keep empty."""

    @pytest.mark.asyncio
    async def test_execution_failure_audit_includes_prior_state(self, fresh_manager):
        """Post-inspection execution failure audit includes prior_state_source
        and bounded_remaining_ttl_seconds."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        _seed_lane_keys(fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[cd_key])

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({cd_key}), cardinality=1,
            ttl_remaining_seconds=123.45,
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=active_inspection,
            ),
            patch(
                f"{_CLEAR_MOD}.clear_cooldown_transaction",
                new_callable=AsyncMock,
                side_effect=RuntimeError("redis down"),
            ),
            patch(f"{_CLEAR_MOD}.logger") as mock_logger,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            assert exc_info.value.status_code == 503

            # Audit must include prior state context.
            payload = mock_logger.info.call_args[0][1]
            assert payload["prior_state_source"] != ""
            assert payload["bounded_remaining_ttl_seconds"] > 0
            assert payload["result"] == "error"
            # Must include resolved candidates/ingress/environment/namespace.
            assert "candidates" in payload
            assert payload["ingress"] == "codex"
            assert "environment" in payload
            assert "namespace" in payload

    @pytest.mark.asyncio
    async def test_postcondition_failure_audit_includes_prior_state(self, fresh_manager):
        """Post-inspection postcondition failure audit includes prior_state_source."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        _seed_lane_keys(fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[cd_key])

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({cd_key}), cardinality=1,
            ttl_remaining_seconds=200.0,
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=active_inspection,
            ),
            patch(
                f"{_CLEAR_MOD}.clear_cooldown_transaction",
                new_callable=AsyncMock,
                return_value=_make_txn_result(),
            ),
            patch(
                f"{_CLEAR_MOD}.verify_aawm_alias_routing_durable_absence",
                new_callable=AsyncMock,
                return_value=False,  # Postcondition failure.
            ),
            patch(
                f"{_CLEAR_MOD}._rollback_committed_results",
                new_callable=AsyncMock,
            ),
            patch(f"{_CLEAR_MOD}.logger") as mock_logger,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            assert exc_info.value.status_code == 500

            payload = mock_logger.info.call_args[0][1]
            assert payload["prior_state_source"] != ""
            assert payload["bounded_remaining_ttl_seconds"] > 0
            assert payload["error_code"] == "postcondition_failure"

    @pytest.mark.asyncio
    async def test_pre_inspection_failure_audit_has_empty_prior(self, fresh_manager):
        """Pre-inspection failures (e.g. auth) keep prior_state_source empty."""
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        with (
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(f"{_CLEAR_MOD}._check_admin_auth", side_effect=HTTPException(
                status_code=403, detail={"error": "forbidden", "message": "no"}
            )),
            patch(f"{_CLEAR_MOD}.logger") as mock_logger,
        ):
            with pytest.raises(HTTPException):
                await handle_cooldown_clear(request, admin)

            payload = mock_logger.info.call_args[0][1]
            assert payload["prior_state_source"] == ""
            assert payload["bounded_remaining_ttl_seconds"] == 0.0

    @pytest.mark.asyncio
    async def test_failure_audit_no_secrets(self, fresh_manager):
        """Post-inspection failure audit never exposes secrets, keys, or hashes."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        _seed_lane_keys(fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[cd_key])

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({cd_key}), cardinality=1,
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=active_inspection,
            ),
            patch(
                f"{_CLEAR_MOD}.clear_cooldown_transaction",
                new_callable=AsyncMock,
                side_effect=RuntimeError("redis down"),
            ),
            patch(f"{_CLEAR_MOD}.logger") as mock_logger,
        ):
            with pytest.raises(HTTPException):
                await handle_cooldown_clear(request, admin)

            payload_str = str(mock_logger.info.call_args)
            assert "sk-" not in payload_str
            assert "Bearer" not in payload_str
            assert cd_key not in payload_str
            assert id_hash not in payload_str


# ===========================================================================
# OpenRouter authoritative atomicity tests
# ===========================================================================


class TestOpenRouterAuthoritativeAtomicity:
    """OpenRouter atomicity: inspect, targeted removal, and post-removal
    absence verification while holding the shared openrouter_rate_limit.lock,
    covering both rate-limit and failure-circuit maps.  For not_active,
    inspect/verify both maps under that same lock immediately before returning.
    A concurrent writer after lock release represents a new upstream failure
    and is allowed, but no mutation observable within the locked critical
    section may survive a cleared/not_active result."""

    @pytest.fixture()
    def fresh_manager(self):
        return AliasRoutingStateManager()

    def _make_openrouter_target(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _ResolvedIdentity,
            _ResolvedTarget,
        )
        ident = _ResolvedIdentity(
            identity_hash="or-atomic-hash",
            provider="openrouter",
            model="openrouter/free",
            route_family="codex_openrouter_responses",
            lane_keys=[],
        )
        return _ResolvedTarget(
            family="codex",
            identities=[ident],
            target_description="alias:test-or-atomic",
            ingress="codex",
        )

    @pytest.mark.asyncio
    async def test_clear_verifies_absence_under_lock(self, fresh_manager):
        """Post-removal absence verification runs under the same lock:
        target keys are provably absent before the lock is released."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _clear_openrouter_local_state,
        )

        mgr = fresh_manager
        now = time.monotonic()
        mgr.openrouter_rate_limit.until_monotonic_by_key["openrouter/free"] = now + 300
        mgr.openrouter_failure_circuit.until_monotonic_by_key["openrouter/free"] = now + 300

        target = self._make_openrouter_target()

        # Track whether keys are absent while lock is held.
        absent_under_lock = False
        original_lock = mgr.openrouter_rate_limit.lock

        class LockSpy:
            """Wraps the real lock to inspect state at release time."""
            def __init__(self, real_lock):
                self._real = real_lock

            async def __aenter__(self):
                await self._real.acquire()
                return self

            async def __aexit__(self, *args):
                nonlocal absent_under_lock
                # At lock release, target keys must be absent.
                absent_under_lock = (
                    "openrouter/free" not in mgr.openrouter_rate_limit.until_monotonic_by_key
                    and "openrouter/free" not in mgr.openrouter_failure_circuit.until_monotonic_by_key
                )
                self._real.release()

        mgr.openrouter_rate_limit.lock = LockSpy(original_lock)

        with patch(
            f"{_CLEAR_MOD}._derive_openrouter_rate_limit_keys",
            return_value=["openrouter/free"],
        ):
            result = await _clear_openrouter_local_state(target, mgr)

        assert result["openrouter_keys_cleared"] == 2
        assert absent_under_lock is True

    @pytest.mark.asyncio
    async def test_inject_during_clear_fails_closed(self, fresh_manager):
        """If a concurrent writer re-creates a target entry during the
        critical section (simulated by injecting between pop and verify),
        the post-removal verification must fail closed."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _clear_openrouter_local_state,
        )

        mgr = fresh_manager
        now = time.monotonic()
        mgr.openrouter_rate_limit.until_monotonic_by_key["openrouter/free"] = now + 300

        target = self._make_openrouter_target()

        # Simulate: after pop but before verify, a concurrent writer
        # re-creates the entry.  We do this by patching the dict's pop
        # to also re-inject.
        original_rl_dict = mgr.openrouter_rate_limit.until_monotonic_by_key
        injected = False

        class InjectingDict(dict):
            def pop(self, key, *args):
                nonlocal injected
                result = super().pop(key, *args)
                if key == "openrouter/free" and not injected:
                    injected = True
                    # Re-inject: simulate concurrent writer.
                    self["openrouter/free"] = time.monotonic() + 999
                return result

        mgr.openrouter_rate_limit.until_monotonic_by_key = InjectingDict(original_rl_dict)

        with (
            patch(
                f"{_CLEAR_MOD}._derive_openrouter_rate_limit_keys",
                return_value=["openrouter/free"],
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _clear_openrouter_local_state(target, mgr)
            assert exc_info.value.status_code == 500
            assert exc_info.value.detail["error"] == "postcondition_failure"

    @pytest.mark.asyncio
    async def test_unrelated_keys_preserved_under_lock(self, fresh_manager):
        """Unrelated keys in both maps are preserved during targeted clear."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _clear_openrouter_local_state,
        )

        mgr = fresh_manager
        now = time.monotonic()
        mgr.openrouter_rate_limit.until_monotonic_by_key["openrouter/free"] = now + 300
        mgr.openrouter_rate_limit.until_monotonic_by_key["unrelated-rl"] = now + 300
        mgr.openrouter_failure_circuit.until_monotonic_by_key["openrouter/free"] = now + 300
        mgr.openrouter_failure_circuit.until_monotonic_by_key["unrelated-fc"] = now + 300

        target = self._make_openrouter_target()

        with patch(
            f"{_CLEAR_MOD}._derive_openrouter_rate_limit_keys",
            return_value=["openrouter/free"],
        ):
            result = await _clear_openrouter_local_state(target, mgr)

        assert result["openrouter_keys_cleared"] == 2
        assert "unrelated-rl" in mgr.openrouter_rate_limit.until_monotonic_by_key
        assert "unrelated-fc" in mgr.openrouter_failure_circuit.until_monotonic_by_key
        assert "openrouter/free" not in mgr.openrouter_rate_limit.until_monotonic_by_key
        assert "openrouter/free" not in mgr.openrouter_failure_circuit.until_monotonic_by_key

    @pytest.mark.asyncio
    async def test_inspect_prior_state_under_lock(self, fresh_manager):
        """_inspect_openrouter_prior_state inspects both maps under the
        shared lock for linearizability."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            _inspect_openrouter_prior_state,
        )

        mgr = fresh_manager
        now = time.monotonic()
        mgr.openrouter_rate_limit.until_monotonic_by_key["openrouter/free"] = now + 300

        target = self._make_openrouter_target()

        original_lock = mgr.openrouter_rate_limit.lock

        class LockTracker:
            def __init__(self, real_lock):
                self._real = real_lock
                self.held = False

            async def __aenter__(self):
                await self._real.acquire()
                self.held = True
                return self

            async def __aexit__(self, *args):
                self.held = False
                self._real.release()

        tracker = LockTracker(original_lock)
        mgr.openrouter_rate_limit.lock = tracker

        with patch(
            f"{_CLEAR_MOD}._derive_openrouter_rate_limit_keys",
            return_value=["openrouter/free"],
        ):
            result = await _inspect_openrouter_prior_state(target, mgr)

        assert result is True
        # Lock was released after inspection.
        assert tracker.held is False

    @pytest.mark.asyncio
    async def test_not_active_verifies_openrouter_under_lock(self, fresh_manager):
        """Full endpoint: not_active path inspects/verifies OpenRouter maps
        under the shared lock immediately before returning."""
        snapshot = _make_snapshot(
            provider="openrouter",
            model="anthropic/claude-3-opus",
            route_family="codex_openrouter_adapter",
        )
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=absent_inspection,
            ),
            patch(
                f"{_CLEAR_MOD}._derive_openrouter_rate_limit_keys",
                return_value=["anthropic/claude-3-opus"],
            ),
        ):
            result = await handle_cooldown_clear(request, admin)
            assert result["result"] == "not_active"

    @pytest.mark.asyncio
    async def test_not_active_fails_closed_on_openrouter_race(self, fresh_manager):
        """If an OpenRouter entry appears between prior-state inspection and
        the not_active lock-held verification, endpoint fails closed (409)."""
        snapshot = _make_snapshot(
            provider="openrouter",
            model="anthropic/claude-3-opus",
            route_family="codex_openrouter_adapter",
        )
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        absent_inspection = _make_identity_inspection(
            exists=False, members=frozenset(), cardinality=0
        )

        # Simulate: prior-state sees no OpenRouter activity, but between
        # that inspection and the not_active lock-held check, a concurrent
        # writer creates an entry.
        call_count = [0]

        async def racing_inspect(target, state_mgr):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call (prior-state): no activity.
                return False
            # Should not be called again in the not_active path because
            # the lock-held check uses inline code, not this function.
            return False

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=absent_inspection,
            ),
            patch(
                f"{_CLEAR_MOD}._derive_openrouter_rate_limit_keys",
                return_value=["anthropic/claude-3-opus"],
            ),
            patch(
                f"{_CLEAR_MOD}._inspect_openrouter_prior_state",
                new_callable=AsyncMock,
                side_effect=racing_inspect,
            ),
        ):
            # Inject the racing entry: it will be present when the not_active
            # lock-held verification runs.
            fresh_manager.openrouter_rate_limit.until_monotonic_by_key[
                "anthropic/claude-3-opus"
            ] = time.monotonic() + 999

            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            assert exc_info.value.status_code == 409
            assert exc_info.value.detail["error"] == "openrouter_state_race"


# ===========================================================================
# Post-inspection failure response enrichment tests
# ===========================================================================


class TestPostInspectionFailureResponseEnrichment:
    """Post-inspection failure responses: when prior source/TTL and trusted
    resolved candidate context are known, the re-raised HTTPException detail
    is enriched with safe prior_state_source, bounded_remaining_ttl_seconds,
    resolved candidates, ingress, environment, and namespace, matching the
    exactly-one audit.  Pre-inspection failures must not invent fields."""

    @pytest.fixture()
    def fresh_manager(self):
        return AliasRoutingStateManager()

    @pytest.mark.asyncio
    async def test_clear_failed_response_enriched(self, fresh_manager):
        """clear_failed response includes prior_state_source, TTL, candidates,
        ingress, environment, and namespace."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        _seed_lane_keys(fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[cd_key])

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({cd_key}), cardinality=1,
            ttl_remaining_seconds=150.0,
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=active_inspection,
            ),
            patch(
                f"{_CLEAR_MOD}.clear_cooldown_transaction",
                new_callable=AsyncMock,
                side_effect=RuntimeError("redis down"),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            assert exc_info.value.status_code == 503
            detail = exc_info.value.detail
            assert detail["error"] == "clear_failed"
            # Enriched fields.
            assert detail["prior_state_source"] != ""
            assert detail["bounded_remaining_ttl_seconds"] > 0
            assert isinstance(detail["candidates"], list)
            assert len(detail["candidates"]) >= 1
            assert detail["candidates"][0]["provider"] == "openai"
            assert detail["ingress"] == "codex"
            assert "environment" in detail
            assert "namespace" in detail
            # Original status/error preserved.
            assert detail["message"] == "durable clear transaction failed; failing closed"

    @pytest.mark.asyncio
    async def test_rollback_failure_response_enriched(self, fresh_manager):
        """rollback_failure response includes enrichment fields."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
            RollbackFailedError,
        )

        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        _seed_lane_keys(fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[cd_key])

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({cd_key}), cardinality=1,
            ttl_remaining_seconds=200.0,
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=active_inspection,
            ),
            patch(
                f"{_CLEAR_MOD}.clear_cooldown_transaction",
                new_callable=AsyncMock,
                side_effect=RollbackFailedError(
                    phase="committed", family="codex",
                    transaction_id_prefix="txn", identity_prefix="id",
                    key_count=1, exception_classes=(),
                ),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            assert exc_info.value.status_code == 503
            detail = exc_info.value.detail
            assert detail["error"] == "rollback_failure"
            assert detail["prior_state_source"] != ""
            assert detail["bounded_remaining_ttl_seconds"] > 0
            assert "candidates" in detail
            assert detail["ingress"] == "codex"

    @pytest.mark.asyncio
    async def test_postcondition_failure_response_enriched(self, fresh_manager):
        """postcondition_failure response includes enrichment fields."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        _seed_lane_keys(fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[cd_key])

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({cd_key}), cardinality=1,
            ttl_remaining_seconds=100.0,
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=active_inspection,
            ),
            patch(
                f"{_CLEAR_MOD}.clear_cooldown_transaction",
                new_callable=AsyncMock,
                return_value=_make_txn_result(),
            ),
            patch(
                f"{_CLEAR_MOD}.verify_aawm_alias_routing_durable_absence",
                new_callable=AsyncMock,
                return_value=False,  # Postcondition failure.
            ),
            patch(
                f"{_CLEAR_MOD}._rollback_committed_results",
                new_callable=AsyncMock,
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            assert exc_info.value.status_code == 500
            detail = exc_info.value.detail
            assert detail["error"] == "postcondition_failure"
            assert detail["prior_state_source"] != ""
            assert detail["bounded_remaining_ttl_seconds"] > 0
            assert "candidates" in detail
            assert detail["ingress"] == "codex"
            assert "environment" in detail
            assert "namespace" in detail

    @pytest.mark.asyncio
    async def test_pre_inspection_failure_not_enriched(self, fresh_manager):
        """Pre-inspection failures (e.g. auth) must NOT invent enrichment fields."""
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        with (
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(f"{_CLEAR_MOD}._check_admin_auth", side_effect=HTTPException(
                status_code=403, detail={"error": "forbidden", "message": "no"}
            )),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            assert exc_info.value.status_code == 403
            detail = exc_info.value.detail
            assert detail["error"] == "forbidden"
            # Must NOT have enrichment fields.
            assert "prior_state_source" not in detail
            assert "bounded_remaining_ttl_seconds" not in detail
            assert "candidates" not in detail

    @pytest.mark.asyncio
    async def test_enriched_response_no_secrets(self, fresh_manager):
        """Enriched failure response never exposes secrets, raw keys, or hashes."""
        snapshot = _make_snapshot()
        request = _make_request({"alias": "test-alias", "ingress": "codex"})
        admin = _make_real_admin_key_dict()

        id_hash = _resolve_identity_hash(
            provider="openai", model="gpt-4o", route_family="codex_openai_responses"
        )
        cd_key = f"h{snapshot.config_hash}:openai:gpt-4o:chatgpt-account:acct1"
        _seed_lane_keys(fresh_manager, family="codex", identity_hash=id_hash, lane_keys=[cd_key])

        active_inspection = _make_identity_inspection(
            exists=True, members=frozenset({cd_key}), cardinality=1,
        )

        with (
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=snapshot),
            patch("litellm.proxy.proxy_server.master_key", "sk-master-secret"),
            patch(f"{_CLEAR_MOD}.alias_routing_state", fresh_manager),
            patch(
                f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache",
                return_value=_make_mock_dual_cache(),
            ),
            patch(
                f"{_CLEAR_MOD}.inspect_identity_set",
                new_callable=AsyncMock,
                return_value=active_inspection,
            ),
            patch(
                f"{_CLEAR_MOD}.clear_cooldown_transaction",
                new_callable=AsyncMock,
                side_effect=RuntimeError("redis down"),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_cooldown_clear(request, admin)
            detail_str = str(exc_info.value.detail)
            assert "sk-" not in detail_str
            assert "Bearer" not in detail_str
            assert cd_key not in detail_str
            assert id_hash not in detail_str
            assert "redis down" not in detail_str
