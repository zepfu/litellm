"""CFG-004 criterion 11: acceptance endpoint unit tests.

Covers: route gates, schema validation, public response fields,
production lane-key equality with _build_codex_auto_agent_candidate_state
for every in-scope provider, prepare/inspect/restore with durable
journal/rollback contracts, restore failure retaining recovery state,
stdin-only secret handling, OAuth-only TUI egress, and exactly one
_cfg003_run_proof_case call.
"""

from __future__ import annotations

import importlib.util
import pathlib
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_acceptance import (
    _check_acceptance_gates,
    _handle_inspect,
    _handle_prepare,
    _handle_restore,
    _prepared_runs,
    _validate_body,
    resolve_production_cooldown_key,
    resolve_production_lane_key,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_snapshot import (
    RoutingAlias,
    RoutingCandidate,
    RoutingSnapshot,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    AliasRoutingStateManager,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MOD = "litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_acceptance"
_SEL_MOD = "litellm.proxy.pass_through_endpoints.aawm_alias_routing.selection"
_RUN_ID = "a" * 32
_NAMESPACE = f"aawm-routing-dev-cfg004-{_RUN_ID}"
_MASTER_KEY = "sk-test-master-key-for-cfg004"
_TARGET_ROUTE_FAMILY = "codex_alibaba_token_plan_chat_completions_adapter"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot() -> RoutingSnapshot:
    target = RoutingCandidate(
        provider="alibaba_token_plan",
        model="alibaba_token_plan/qwen3.6-flash",
        route_family=_TARGET_ROUTE_FAMILY,
        priority=10,
        weight=1.0,
        tui_attached=None,
        schedule=None,
    )
    control = RoutingCandidate(
        provider="openai",
        model="gpt-5.3-codex",
        route_family="codex_openai_responses",
        priority=20,
        weight=1.0,
        tui_attached=None,
        schedule=None,
    )
    alias = RoutingAlias(
        name="basic",
        distribution_strategy=None,
        candidates=(target, control),
    )
    return RoutingSnapshot(
        aliases={"basic": alias},
        config_epoch=1,
        config_hash="abcdef123456",
        config_version="abcdef123456",
    )


def _make_request(*, auth_header: str = "Bearer sk-test-auth") -> MagicMock:
    """Build a mock Request with headers for lane-key derivation."""
    request = MagicMock()
    request.headers = {"authorization": auth_header}
    return request


def _make_admin_key_dict(*, api_key: str = "sk-master-secret") -> MagicMock:
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

    mock = MagicMock()
    mock.user_role = LitellmUserRoles.PROXY_ADMIN
    mock.token = UserAPIKeyAuth._safe_hash_litellm_api_key(api_key)
    return mock


def _gate_env(run_id: str = _RUN_ID) -> dict[str, str]:
    return {
        "AAWM_LITELLM_ENVIRONMENT": "litellm-dev",
        "AAWM_CFG004_ACCEPTANCE_ENABLED": "1",
        "AAWM_CFG004_ACCEPTANCE_RUN_ID": run_id,
        "AAWM_ALIAS_ROUTING_STATE_NAMESPACE": f"aawm-routing-dev-cfg004-{run_id}",
        "AAWM_ALIAS_ROUTING_COOLDOWN_CLEAR_SINGLE_WORKER": "1",
    }


def _mock_proxy_server():
    mock = MagicMock()
    mock.master_key = _MASTER_KEY
    return mock


def _mock_dual_cache():
    """Return a non-None DualCache mock with redis_cache."""
    dc = MagicMock()
    dc.redis_cache = MagicMock()
    dc.redis_cache.init_async_client = MagicMock(return_value=MagicMock())
    return dc


def _mock_txn_result(identity_hash: str = "ih", lane_key: str = "lk"):
    """Build a mock CooldownTransactionResult with journal."""
    journal = MagicMock()
    journal.transaction_id = "txn123"
    journal.phase = "DURABLE_COMMITTED"
    journal.alias_family = "codex"
    journal.identity_hash = identity_hash
    journal.cooldown_keys = [lane_key]
    journal.identity_keys = [f"id:{identity_hash}"]
    journal.lane_members = [lane_key]
    journal.preimages = [(None, -2)]
    journal.receipt_key = f"receipt:{identity_hash}"
    journal.requested_ttl = 300
    result = MagicMock()
    result.transaction_id = "txn123"
    result.phase = "DURABLE_COMMITTED"
    result.journal = journal
    return result


def _mock_inspection(*, exists: bool = False, cardinality: int = 0):
    insp = MagicMock()
    insp.exists = exists
    insp.cardinality = cardinality
    return insp


@pytest.fixture(autouse=True)
def _clear_prepared():
    _prepared_runs.clear()
    yield
    _prepared_runs.clear()


_LANE_KEYS_MOD = "litellm.proxy.pass_through_endpoints.aawm_alias_routing.lane_keys"


def _fake_safe_get_request_headers(request):
    """Minimal stand-in for the runtime-injected _safe_get_request_headers."""
    if request is None:
        return {}
    headers = getattr(request, "headers", None)
    if isinstance(headers, dict):
        return headers
    return {}


def _fake_clean_codex_auth_value(value):
    """Minimal stand-in for _clean_codex_auth_value."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.lower().startswith("bearer "):
        s = s[7:].strip()
    return s or None


@pytest.fixture(autouse=True)
def _patch_lane_keys_headers():
    """Inject host-globals into lane_keys (normally done by install())."""
    import litellm.proxy.pass_through_endpoints.aawm_alias_routing.lane_keys as lk_mod
    originals = {}
    for name, val in [
        ("_safe_get_request_headers", _fake_safe_get_request_headers),
        ("_clean_codex_auth_value", _fake_clean_codex_auth_value),
    ]:
        originals[name] = getattr(lk_mod, name, None)
        setattr(lk_mod, name, val)
    yield
    for name, orig in originals.items():
        if orig is not None:
            setattr(lk_mod, name, orig)
        else:
            try:
                delattr(lk_mod, name)
            except AttributeError:
                pass


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_valid_prepare_body(self):
        op, rid = _validate_body({
            "operation": "prepare", "run_id": _RUN_ID, "alias": "basic",
            "ingress": "codex", "provider": "alibaba_token_plan",
            "model": "alibaba_token_plan/qwen3.6-flash", "ttl_seconds": 300,
        })
        assert op == "prepare"
        assert rid == _RUN_ID

    def test_valid_inspect_body(self):
        op, _ = _validate_body({"operation": "inspect", "run_id": _RUN_ID})
        assert op == "inspect"

    def test_valid_restore_body(self):
        op, _ = _validate_body({"operation": "restore", "run_id": _RUN_ID})
        assert op == "restore"

    def test_invalid_operation(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_body({"operation": "destroy", "run_id": _RUN_ID})
        assert exc_info.value.status_code == 400

    def test_invalid_run_id_short(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_body({"operation": "inspect", "run_id": "abc"})
        assert exc_info.value.status_code == 400

    def test_extra_fields_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_body({
                "operation": "prepare", "run_id": _RUN_ID, "alias": "basic",
                "ingress": "codex", "provider": "x", "model": "y",
                "ttl_seconds": 60, "extra": "bad",
            })
        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# Gate tests
# ---------------------------------------------------------------------------


class TestGates:
    @patch.dict("os.environ", _gate_env(), clear=False)
    @patch(f"{_MOD}.get_aawm_alias_routing_state_namespace", return_value=_NAMESPACE)
    def test_all_gates_pass(self, _mock_ns):
        _check_acceptance_gates(_RUN_ID)

    @patch.dict("os.environ", {**_gate_env(), "AAWM_LITELLM_ENVIRONMENT": "prod"}, clear=False)
    @patch(f"{_MOD}.get_aawm_alias_routing_state_namespace", return_value=_NAMESPACE)
    def test_wrong_environment(self, _mock_ns):
        with pytest.raises(HTTPException) as exc_info:
            _check_acceptance_gates(_RUN_ID)
        assert exc_info.value.status_code == 503

    @patch.dict("os.environ", _gate_env(), clear=False)
    @patch(f"{_MOD}.get_aawm_alias_routing_state_namespace", return_value="wrong-ns")
    def test_namespace_mismatch(self, _mock_ns):
        with pytest.raises(HTTPException) as exc_info:
            _check_acceptance_gates(_RUN_ID)
        assert exc_info.value.status_code == 503

    @patch.dict("os.environ", _gate_env(run_id="b" * 32), clear=False)
    @patch(f"{_MOD}.get_aawm_alias_routing_state_namespace", return_value=_NAMESPACE)
    def test_run_id_mismatch(self, _mock_ns):
        with pytest.raises(HTTPException) as exc_info:
            _check_acceptance_gates(_RUN_ID)
        assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# Production lane-key equality: acceptance vs _build_codex_auto_agent_candidate_state
# ---------------------------------------------------------------------------


class TestProductionLaneKeyEquality:
    """Prove acceptance keys equal production selector keys for every
    in-scope provider family."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider,model,route_family", [
        ("openrouter", "openrouter/some-model", "codex_openrouter_responses"),
        ("kimi_code", "kimi_code/kimi-model", "codex_kimi_code_responses"),
        ("alibaba_token_plan", "alibaba_token_plan/qwen3.6-flash", _TARGET_ROUTE_FAMILY),
        ("opencode_zen", "opencode_zen/zen-model", "codex_opencode_responses"),
        ("openai", "gpt-5.3-codex", "codex_openai_responses"),
    ])
    async def test_cooldown_key_matches_production(
        self, provider, model, route_family
    ):
        """For each in-scope provider, resolve_production_cooldown_key must
        equal the cooldown_key from _build_codex_auto_agent_candidate_state."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.selection import (
            _build_codex_auto_agent_candidate_state,
        )

        candidate = {
            "provider": provider,
            "model": model,
            "route_family": route_family,
            "config_epoch_tag": "abcdef123456",
        }
        request = _make_request(auth_header="Bearer sk-test-auth-value")

        # Acceptance derivation.
        acceptance_key = resolve_production_cooldown_key(request, candidate)

        # Production derivation: mock the injected cooldown-state reader and
        # all _apply_* filters to be passthrough so only the lane/key logic runs.
        async def _fake_get_state(cooldown_key):
            return (0.0, "none")

        async def _fake_grok(*, candidate, lane_key, cooldown_seconds, cooldown_state_source, skip_reason, get_active_cooldown_state):
            return (cooldown_seconds, cooldown_state_source, skip_reason)

        async def _fake_kimi(*, candidate, cooldown_seconds, cooldown_state_source, skip_reason, get_active_cooldown_state):
            return (cooldown_seconds, cooldown_state_source, skip_reason, None)

        def _fake_request_local(request, *, candidate, lane_key, cooldown_seconds, cooldown_state_source, skip_reason):
            return (cooldown_seconds, cooldown_state_source, skip_reason)

        async def _fake_adapter(*, candidate, cooldown_seconds, cooldown_state_source, skip_reason):
            return (cooldown_seconds, cooldown_state_source, skip_reason)

        async def _fake_openrouter(*, candidate, cooldown_seconds, cooldown_state_source, skip_reason):
            return (cooldown_seconds, cooldown_state_source, skip_reason)

        with (
            patch(f"{_SEL_MOD}._get_codex_active_cooldown_state", _fake_get_state),
            patch(f"{_SEL_MOD}._apply_codex_auto_agent_grok_account_lane_cooldown", _fake_grok),
            patch(f"{_SEL_MOD}._apply_kimi_code_managed_account_lane_cooldown", _fake_kimi),
            patch(f"{_SEL_MOD}._apply_codex_auto_agent_request_local_candidate_state", _fake_request_local),
            patch(f"{_SEL_MOD}._apply_codex_auto_agent_adapter_local_candidate_cooldown", _fake_adapter),
            patch(f"{_SEL_MOD}._apply_openrouter_durable_quota_candidate_cooldown", _fake_openrouter),
        ):
            state = await _build_codex_auto_agent_candidate_state(
                request,
                candidate_template=candidate,
            )

        assert acceptance_key == state["cooldown_key"], (
            f"provider={provider}: acceptance={acceptance_key!r} != production={state['cooldown_key']!r}"
        )

    def test_lane_key_provider_branches(self):
        """Verify the provider-aware branch selects correct lane constants."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.policy import (
            CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY,
            CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY,
            CODEX_AUTO_AGENT_OPENCODE_LANE_KEY,
            CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY,
        )

        request = _make_request()
        assert resolve_production_lane_key(request, {"provider": "openrouter"}) == CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY
        assert resolve_production_lane_key(request, {"provider": "kimi_code"}) == CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY
        assert resolve_production_lane_key(request, {"provider": "alibaba_token_plan"}) == CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY
        assert resolve_production_lane_key(request, {"provider": "opencode_zen"}) == CODEX_AUTO_AGENT_OPENCODE_LANE_KEY
        # OpenAI/native: derives from request auth header.
        openai_lane = resolve_production_lane_key(request, {"provider": "openai"})
        assert openai_lane.startswith("auth:")


# ---------------------------------------------------------------------------
# Prepare / Inspect / Restore with durable contracts
# ---------------------------------------------------------------------------


class TestPrepareInspectRestore:
    @pytest.fixture
    def state_mgr(self):
        return AliasRoutingStateManager()

    @pytest.mark.asyncio
    @patch(f"{_MOD}.rollback_cooldown_transaction", new_callable=AsyncMock)
    @patch(f"{_MOD}.publish_cooldown_transaction", new_callable=AsyncMock)
    @patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
    @patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
    @patch("litellm.proxy.proxy_server", new_callable=_mock_proxy_server, create=True)
    async def test_prepare_seeds_and_stores_journals(
        self, _ps, _snap, mock_dc, mock_inspect, mock_publish, mock_rollback, state_mgr
    ):
        mock_dc.return_value = _mock_dual_cache()
        # prestate: 2 absent, post-seed verify: 2 present
        mock_inspect.side_effect = [
            _mock_inspection(exists=False, cardinality=0),
            _mock_inspection(exists=False, cardinality=0),
            _mock_inspection(exists=True, cardinality=1),
            _mock_inspection(exists=True, cardinality=1),
        ]
        mock_publish.return_value = _mock_txn_result()

        request = _make_request()
        body = {
            "operation": "prepare", "run_id": _RUN_ID, "alias": "basic",
            "ingress": "codex", "provider": "alibaba_token_plan",
            "model": "alibaba_token_plan/qwen3.6-flash", "ttl_seconds": 300,
        }
        result = await _handle_prepare(body, _RUN_ID, request, state_mgr)
        assert result["result"] == "prepared"
        assert result["control_count"] == 1
        assert _RUN_ID in _prepared_runs
        # Journals retained.
        prepared = _prepared_runs[_RUN_ID]
        assert len(prepared.publication_results) == 2  # 1 control + 1 target
        assert prepared.target_publication_result is not None
        assert len(prepared.control_publication_results) == 1

    @pytest.mark.asyncio
    @patch(f"{_MOD}.get_aawm_alias_routing_dual_cache", return_value=None)
    @patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
    @patch("litellm.proxy.proxy_server", new_callable=_mock_proxy_server, create=True)
    async def test_prepare_fails_503_without_dual_cache(self, _ps, _snap, mock_dc, state_mgr):
        """No local-only acceptance: fail 503 when DualCache is None."""
        request = _make_request()
        body = {
            "operation": "prepare", "run_id": _RUN_ID, "alias": "basic",
            "ingress": "codex", "provider": "alibaba_token_plan",
            "model": "alibaba_token_plan/qwen3.6-flash", "ttl_seconds": 300,
        }
        with pytest.raises(HTTPException) as exc_info:
            await _handle_prepare(body, _RUN_ID, request, state_mgr)
        assert exc_info.value.status_code == 503
        assert "redis_unavailable" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    @patch(f"{_MOD}.rollback_cooldown_transaction", new_callable=AsyncMock)
    @patch(f"{_MOD}.publish_cooldown_transaction", new_callable=AsyncMock)
    @patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
    @patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
    @patch("litellm.proxy.proxy_server", new_callable=_mock_proxy_server, create=True)
    async def test_prepare_rollback_on_publish_failure(
        self, _ps, _snap, mock_dc, mock_inspect, mock_publish, mock_rollback, state_mgr
    ):
        """On publish failure, reverse-order rollback is invoked."""
        mock_dc.return_value = _mock_dual_cache()
        mock_inspect.return_value = _mock_inspection(exists=False, cardinality=0)
        # First publish succeeds, second fails.
        mock_publish.side_effect = [_mock_txn_result(), RuntimeError("publish failed")]

        request = _make_request()
        body = {
            "operation": "prepare", "run_id": _RUN_ID, "alias": "basic",
            "ingress": "codex", "provider": "alibaba_token_plan",
            "model": "alibaba_token_plan/qwen3.6-flash", "ttl_seconds": 300,
        }
        with pytest.raises(HTTPException) as exc_info:
            await _handle_prepare(body, _RUN_ID, request, state_mgr)
        assert exc_info.value.status_code == 500
        # Rollback called for the one committed publication.
        mock_rollback.assert_called_once()
        # Recovery record retained for operator inspection (defect 2).
        assert _RUN_ID in _prepared_runs

    @pytest.mark.asyncio
    @patch(f"{_MOD}.verify_aawm_alias_routing_durable_absence", new_callable=AsyncMock)
    @patch(f"{_MOD}.delete_aawm_alias_routing_durable_key", new_callable=AsyncMock)
    @patch(f"{_MOD}.rollback_cooldown_transaction", new_callable=AsyncMock)
    @patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
    @patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
    @patch("litellm.proxy.proxy_server", new_callable=_mock_proxy_server, create=True)
    async def test_restore_verifies_absence_and_pops(
        self, _ps, _snap, mock_dc, mock_inspect, mock_rollback, mock_delete, mock_verify, state_mgr
    ):
        """Restore verifies local+durable absence, then pops."""
        mock_dc.return_value = _mock_dual_cache()
        mock_inspect.return_value = _mock_inspection(exists=False, cardinality=0)
        mock_verify.return_value = True
        mock_delete.return_value = True

        # Seed prepared state manually.
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_acceptance import _PreparedState
        _prepared_runs[_RUN_ID] = _PreparedState(
            run_id=_RUN_ID,
            target_identity_hash="tih",
            target_lane_key="tlk",
            control_identity_hashes=["cih"],
            control_lane_keys=["clk"],
            all_identity_hashes=["tih", "cih"],
            all_lane_keys=["tlk", "clk"],
            publication_results=[_mock_txn_result()],
            target_publication_result=_mock_txn_result(),
            control_publication_results=[_mock_txn_result()],
            local_preimages={},
            prepared_at=time.monotonic(),
        )

        result = await _handle_restore(_RUN_ID, state_mgr)
        assert result["result"] == "restored"
        assert result["cleared_identities"] == 2
        assert _RUN_ID not in _prepared_runs
        # Target receipt deleted via targeted key deletion (not rollback).
        mock_delete.assert_called_once_with(
            alias_family="codex",
            state_kind="txn_receipt",
            state_key="txn-receipt:txn123",
        )
        # Control rollback called exactly once (strict reverse-order).
        mock_rollback.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{_MOD}.verify_aawm_alias_routing_durable_absence", new_callable=AsyncMock)
    @patch(f"{_MOD}.rollback_cooldown_transaction", new_callable=AsyncMock)
    @patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
    @patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
    @patch("litellm.proxy.proxy_server", new_callable=_mock_proxy_server, create=True)
    async def test_restore_failure_retains_recovery_state(
        self, _ps, _snap, mock_dc, mock_inspect, mock_rollback, mock_verify, state_mgr
    ):
        """When verification fails, _prepared_runs is retained."""
        mock_dc.return_value = _mock_dual_cache()
        # Identity still has members -> verification fails.
        mock_inspect.return_value = _mock_inspection(exists=True, cardinality=1)
        mock_verify.return_value = True

        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_acceptance import _PreparedState
        _prepared_runs[_RUN_ID] = _PreparedState(
            run_id=_RUN_ID,
            target_identity_hash="tih",
            target_lane_key="tlk",
            control_identity_hashes=["cih"],
            control_lane_keys=["clk"],
            all_identity_hashes=["tih", "cih"],
            all_lane_keys=["tlk", "clk"],
            publication_results=[],
            control_publication_results=[],
            local_preimages={},
            prepared_at=time.monotonic(),
        )

        with pytest.raises(HTTPException) as exc_info:
            await _handle_restore(_RUN_ID, state_mgr)
        assert exc_info.value.status_code == 500
        assert "target_absence_not_proven" in str(exc_info.value.detail)
        # Recovery state retained.
        assert _RUN_ID in _prepared_runs

    @pytest.mark.asyncio
    @patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
    @patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
    @patch("litellm.proxy.proxy_server", new_callable=_mock_proxy_server, create=True)
    async def test_inspect_not_prepared_returns_404(
        self, _ps, _snap, mock_dc, mock_inspect, state_mgr
    ):
        mock_dc.return_value = _mock_dual_cache()
        request = _make_request()
        with pytest.raises(HTTPException) as exc_info:
            await _handle_inspect(_RUN_ID, request, state_mgr)
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Public field / no-secret tests
# ---------------------------------------------------------------------------


class TestPublicFields:
    @pytest.mark.asyncio
    @patch(f"{_MOD}.rollback_cooldown_transaction", new_callable=AsyncMock)
    @patch(f"{_MOD}.publish_cooldown_transaction", new_callable=AsyncMock)
    @patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
    @patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
    @patch("litellm.proxy.proxy_server", new_callable=_mock_proxy_server, create=True)
    async def test_prepare_response_no_secrets(
        self, _ps, _snap, mock_dc, mock_inspect, mock_publish, mock_rollback
    ):
        mock_dc.return_value = _mock_dual_cache()
        mock_inspect.side_effect = [
            _mock_inspection(exists=False, cardinality=0),
            _mock_inspection(exists=False, cardinality=0),
            _mock_inspection(exists=True, cardinality=1),
            _mock_inspection(exists=True, cardinality=1),
        ]
        mock_publish.return_value = _mock_txn_result()

        state_mgr = AliasRoutingStateManager()
        request = _make_request()
        body = {
            "operation": "prepare", "run_id": _RUN_ID, "alias": "basic",
            "ingress": "codex", "provider": "alibaba_token_plan",
            "model": "alibaba_token_plan/qwen3.6-flash", "ttl_seconds": 300,
        }
        result = await _handle_prepare(body, _RUN_ID, request, state_mgr)
        result_str = str(result)
        assert "identity_hash" not in result_str
        assert "lane_key" not in result_str
        assert _MASTER_KEY not in result_str


# ---------------------------------------------------------------------------
# TTL boundary tests
# ---------------------------------------------------------------------------


class TestTTLBounds:
    @pytest.mark.asyncio
    @patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
    @patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
    @patch("litellm.proxy.proxy_server", new_callable=_mock_proxy_server, create=True)
    async def test_ttl_too_low(self, _ps, _snap, mock_dc):
        mock_dc.return_value = _mock_dual_cache()
        state_mgr = AliasRoutingStateManager()
        request = _make_request()
        body = {
            "operation": "prepare", "run_id": _RUN_ID, "alias": "basic",
            "ingress": "codex", "provider": "alibaba_token_plan",
            "model": "alibaba_token_plan/qwen3.6-flash", "ttl_seconds": 5,
        }
        with pytest.raises(HTTPException) as exc_info:
            await _handle_prepare(body, _RUN_ID, request, state_mgr)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    @patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
    @patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
    @patch("litellm.proxy.proxy_server", new_callable=_mock_proxy_server, create=True)
    async def test_ttl_too_high(self, _ps, _snap, mock_dc):
        mock_dc.return_value = _mock_dual_cache()
        state_mgr = AliasRoutingStateManager()
        request = _make_request()
        body = {
            "operation": "prepare", "run_id": _RUN_ID, "alias": "basic",
            "ingress": "codex", "provider": "alibaba_token_plan",
            "model": "alibaba_token_plan/qwen3.6-flash", "ttl_seconds": 9999,
        }
        with pytest.raises(HTTPException) as exc_info:
            await _handle_prepare(body, _RUN_ID, request, state_mgr)
        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# Acceptance script: stdin-only secret, OAuth-only TUI egress, one proof case
# ---------------------------------------------------------------------------

_SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[4]
    / "scripts" / "local-ci" / "run_anthropic_adapter_acceptance.py"
)


def _load_acceptance_script():
    spec = importlib.util.spec_from_file_location(
        "run_anthropic_adapter_acceptance", _SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestAcceptanceScriptSecrets:
    @pytest.fixture(autouse=True)
    def _load_script(self):
        self.script = _load_acceptance_script()

    def test_compose_override_contains_env_vars(self):
        yaml_str = self.script._cfg004_compose_override_yaml(
            master_key="sk-test", run_id="f" * 32,
            langfuse_public_key="pk-lf-test", langfuse_secret_key="sk-lf-test",
        )
        assert "LITELLM_MASTER_KEY=sk-test" in yaml_str
        assert f"AAWM_CFG004_ACCEPTANCE_RUN_ID={'f' * 32}" in yaml_str
        assert "AAWM_CFG004_ACCEPTANCE_ENABLED=1" in yaml_str
        assert "LANGFUSE_PUBLIC_KEY=pk-lf-test" in yaml_str
        assert "LANGFUSE_SECRET_KEY=sk-lf-test" in yaml_str

    def test_compose_up_passes_override_via_stdin(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr=b"")
            ok, _ = self.script._cfg004_docker_compose_up(override_stdin="services:\n")
        assert ok
        assert mock_run.call_args.kwargs.get("input") is not None

    def test_compose_up_base_has_no_stdin(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr=b"")
            ok, _ = self.script._cfg004_docker_compose_up(override_stdin=None)
        assert ok
        assert mock_run.call_args.kwargs.get("input") is None

    def test_exactly_one_proof_case_call_no_api_key_to_tui(self):
        proof_call_count = 0
        captured_cases = {}

        def _fake_proof_case(**kwargs):
            nonlocal proof_call_count, captured_cases
            proof_call_count += 1
            captured_cases = kwargs.get("cases", {})
            return {
                "result": {
                    "passed": True,
                    "langfuse": {"command_thread_id": "test-thread-id"},
                },
                "selection": {
                    "provider": "alibaba_token_plan",
                    "model": "alibaba_token_plan/qwen3.6-flash",
                    "route_family": "codex_alibaba_token_plan_chat_completions_adapter",
                },
                "phase_session_id": "test-session",
                "phase_start_time": "2026-01-01T00:00:00Z",
            }

        cases = {
            "native_openai_passthrough_responses_codex_basic_alias_collaboration": {
                "verification_alias": "basic",
            }
        }
        prepare_resp = {
            "result": "prepared",
            "target": {"provider": "alibaba_token_plan", "model": "alibaba_token_plan/qwen3.6-flash"},
            "control_count": 1,
            "controls": [{"provider": "openai", "model": "gpt-5.3-codex", "route_family": "codex_openai_responses"}],
        }
        inspect_all_active = {
            "result": "inspected",
            "candidates": [
                {"provider": "alibaba_token_plan", "model": "alibaba_token_plan/qwen3.6-flash",
                 "route_family": "codex_alibaba_token_plan_chat_completions_adapter",
                 "role": "target", "local_cooldown_active": True, "durable_cooldown_active": True},
                {"provider": "openai", "model": "gpt-5.3-codex",
                 "route_family": "codex_openai_responses",
                 "role": "control", "local_cooldown_active": True, "durable_cooldown_active": True},
            ],
        }
        inspect_target_cleared = {
            "result": "inspected",
            "candidates": [
                {"provider": "alibaba_token_plan", "model": "alibaba_token_plan/qwen3.6-flash",
                 "route_family": "codex_alibaba_token_plan_chat_completions_adapter",
                 "role": "target", "local_cooldown_active": False, "durable_cooldown_active": False},
                {"provider": "openai", "model": "gpt-5.3-codex",
                 "route_family": "codex_openai_responses",
                 "role": "control", "local_cooldown_active": True, "durable_cooldown_active": True},
            ],
        }

        with (
            patch.object(self.script, "_cfg004_docker_compose_up", return_value=(True, "")),
            patch.object(self.script, "_cfg004_wait_readiness", return_value=(True, "")),
            patch.object(self.script, "_cfg004_build_disposable_config", return_value="/tmp/cfg004-disposable.yaml"),
            patch.object(self.script, "_cfg004_remove_disposable_config"),
            patch.object(self.script, "_cfg004_readiness_snapshot", return_value={
                "status": 200, "config_hash": "abc", "config_version": "abc", "state": "active",
            }),
            patch.object(self.script, "_cfg004_docker_inspect_container", return_value={
                "id": "x", "image": "y", "started_at": "z", "restart_count": 0,
            }),
            patch.object(self.script, "_cfg004_docker_inspect_env_absent", return_value=[]),
            patch.object(self.script, "_cfg004_post_acceptance", side_effect=[
                (200, prepare_resp),
                (200, inspect_all_active),
                (200, inspect_target_cleared),
                (200, inspect_target_cleared),  # post-proof
                (200, {"result": "restored", "cleared_identities": 2}),
            ]),
            patch.object(self.script, "_cfg004_post_clear", return_value=(
                200, {"result": "cleared", "keys_cleared": 1, "members_removed": 1}
            )),
            patch.object(self.script, "_cfg003_run_proof_case", side_effect=_fake_proof_case),
            patch.object(
                self.script,
                "_cfg004_validate_transcript_collaboration",
                return_value=({"enabled": True, "thread_id": "test-thread-id", "total_child_commands": 3}, []),
            ),
            patch.object(self.script, "RA") as mock_ra,
        ):
            mock_ra._redact_sensitive_artifact_fields = lambda x: x
            mock_ra._HEALTH_READINESS_PATH = "/health/readiness"
            result = self.script._cfg004_cooldown_clear_live_test(
                litellm_base_url="http://127.0.0.1:4001",
                cases=cases,
                suite_config={},
                query_url="http://localhost:3000",
                public_key="pk-test",
                secret_key="sk-test",
            )

        assert proof_call_count == 1
        assert result["passed"] is True
        # No LiteLLM/proxy API key reaches the TUI child.  The TUI retains
        # its existing resolved provider (requires_openai_auth=true) and uses
        # its normal Codex OAuth Authorization credential.
        proof_key = "native_openai_passthrough_responses_codex_basic_alias_collaboration"
        proof_env = captured_cases[proof_key].get("env", {})
        assert "CFG004_PROXY_API_KEY" not in proof_env
        assert "OPENAI_API_KEY" not in proof_env
        assert not any(
            str(v).startswith("sk-cfg004-acceptance-") for v in proof_env.values()
        )
        # Command retains requires_openai_auth=true (no env_key/false override).
        cmd = captured_cases[proof_key].get("command", [])
        cmd_str = " ".join(str(c) for c in cmd)
        assert "requires_openai_auth=false" not in cmd_str
        assert "env_key=" not in cmd_str

    def test_oauth_only_child_env_suppresses_parent_openai_api_key(self, monkeypatch):
        """Even with a sentinel parent OPENAI_API_KEY, the effective child env
        passed to the command lacks it, and the parent env is restored after
        both success and exception."""
        import os

        sentinel = "sk-parent-sentinel-openai-key"
        monkeypatch.setenv("OPENAI_API_KEY", sentinel)
        # Effective child env captured at the moment the proof runs, i.e. what
        # _scrubbed_child_env would build for the TUI subprocess.
        captured_child_env: dict[str, str] = {}

        cases = {
            "native_openai_passthrough_responses_codex_basic_alias_collaboration": {
                "verification_alias": "basic",
                "command": ["codex", "exec", "hi"],
            }
        }

        def _run_live(proof_side_effect):
            inspect_active = {
                "result": "inspected",
                "candidates": [
                    {"provider": "alibaba_token_plan", "model": "alibaba_token_plan/qwen3.6-flash",
                     "route_family": "codex_alibaba_token_plan_chat_completions_adapter",
                     "role": "target", "local_cooldown_active": True, "durable_cooldown_active": True},
                    {"provider": "openai", "model": "gpt-5.3-codex",
                     "route_family": "codex_openai_responses",
                    "role": "control", "local_cooldown_active": True, "durable_cooldown_active": True},
                ],
            }
            inspect_target_cleared = {
                "result": "inspected",
                "candidates": [
                    {"provider": "alibaba_token_plan", "model": "alibaba_token_plan/qwen3.6-flash",
                     "route_family": "codex_alibaba_token_plan_chat_completions_adapter",
                     "role": "target", "local_cooldown_active": False, "durable_cooldown_active": False},
                    {"provider": "openai", "model": "gpt-5.3-codex",
                     "route_family": "codex_openai_responses",
                     "role": "control", "local_cooldown_active": True, "durable_cooldown_active": True},
                ],
            }
            prepare_resp = {
                "result": "prepared",
                "target": {"provider": "alibaba_token_plan", "model": "alibaba_token_plan/qwen3.6-flash"},
                "control_count": 1,
                "controls": [{"provider": "openai", "model": "gpt-5.3-codex", "route_family": "codex_openai_responses"}],
            }
            with (
                patch.object(self.script, "_cfg004_docker_compose_up", return_value=(True, "")),
                patch.object(self.script, "_cfg004_wait_readiness", return_value=(True, "")),
                patch.object(self.script, "_cfg004_build_disposable_config", return_value="/tmp/x.yaml"),
                patch.object(self.script, "_cfg004_remove_disposable_config"),
                patch.object(self.script, "_cfg004_readiness_snapshot", return_value={
                    "status": 200, "config_hash": "abc", "config_version": "abc", "state": "active",
                }),
                patch.object(self.script, "_cfg004_docker_inspect_container", return_value={
                    "id": "x", "image": "y", "started_at": "z", "restart_count": 0,
                }),
                patch.object(self.script, "_cfg004_docker_inspect_env_absent", return_value=[]),
                patch.object(self.script, "_cfg004_post_acceptance", side_effect=[
                    (200, prepare_resp),
                    (200, inspect_active),
                    (200, inspect_target_cleared),
                    (200, inspect_target_cleared),
                    (200, {"result": "restored", "cleared_identities": 2}),
                ]),
                patch.object(self.script, "_cfg004_post_clear", return_value=(
                    200, {"result": "cleared", "keys_cleared": 1, "members_removed": 1}
                )),
                patch.object(self.script, "_cfg003_run_proof_case", side_effect=proof_side_effect),
                patch.object(
                    self.script,
                    "_cfg004_validate_transcript_collaboration",
                    return_value=({"enabled": True, "thread_id": "test-thread-id", "total_child_commands": 3}, []),
                ),
            ):
                return self.script._cfg004_cooldown_clear_live_test(
                    litellm_base_url="http://127.0.0.1:4001",
                    cases=cases,
                    suite_config={},
                    query_url="http://localhost:3000",
                    public_key="pk-test",
                    secret_key="sk-test",
                )

        # Success path: child env lacks the sentinel; parent restored.
        def _ok_proof(**kwargs):
            captured_child_env.update(self.script.RA._scrubbed_child_env(None))
            return {
                "result": {
                    "passed": True,
                    "langfuse": {"command_thread_id": "test-thread-id"},
                },
                "selection": {
                    "provider": "alibaba_token_plan",
                    "model": "alibaba_token_plan/qwen3.6-flash",
                    "route_family": "codex_alibaba_token_plan_chat_completions_adapter",
                },
                "phase_session_id": "s",
                "phase_start_time": "t",
            }

        result = _run_live(_ok_proof)
        assert result["passed"] is True
        assert "OPENAI_API_KEY" not in captured_child_env
        assert os.environ.get("OPENAI_API_KEY") == sentinel

        # Exception path: parent env still restored.
        def _boom_proof(**kwargs):
            captured_child_env.update(self.script.RA._scrubbed_child_env(None))
            raise RuntimeError("proof exploded")

        with pytest.raises(RuntimeError):
            _run_live(_boom_proof)
        assert "OPENAI_API_KEY" not in captured_child_env
        assert os.environ.get("OPENAI_API_KEY") == sentinel

    def test_disposable_config_adds_only_oauth_public_route(self, tmp_path):
        """Disposable config derives from canonical and adds only the
        /openai_passthrough/* public route."""
        import yaml as _yaml

        canonical = {
            "model_list": [{"model_name": "m"}],
            "general_settings": {"aawm_route_use_x_forwarded_for": True},
        }
        canonical_path = tmp_path / "litellm-dev-config.yaml"
        canonical_path.write_text(_yaml.safe_dump(canonical), encoding="utf-8")
        disposable_path = tmp_path / "litellm-dev-config.cfg004-acceptance.yaml"
        with (
            patch.object(self.script, "_CFG004_DEV_CONFIG_PATH", canonical_path),
            patch.object(self.script, "_CFG004_DISPOSABLE_CONFIG_PATH", disposable_path),
        ):
            result_path = self.script._cfg004_build_disposable_config()
        assert result_path == str(disposable_path)
        built = _yaml.safe_load(disposable_path.read_text(encoding="utf-8"))
        # Only addition is the OAuth pass-through public route.
        assert built["general_settings"]["public_routes"] == ["/openai_passthrough/*"]
        # Canonical structure preserved.
        assert built["model_list"] == [{"model_name": "m"}]
        assert built["general_settings"]["aawm_route_use_x_forwarded_for"] is True
        # Canonical file untouched.
        assert "public_routes" not in _yaml.safe_load(
            canonical_path.read_text(encoding="utf-8")
        )["general_settings"]

    def test_compose_override_mounts_disposable_config(self):
        yaml_str = self.script._cfg004_compose_override_yaml(
            master_key="sk-test", run_id="f" * 32,
            disposable_config_host_path="/host/cfg004-disposable.yaml",
        )
        assert "/host/cfg004-disposable.yaml:/app/litellm-dev-config.yaml" in yaml_str

    def test_compose_override_no_mount_without_disposable_config(self):
        yaml_str = self.script._cfg004_compose_override_yaml(
            master_key="sk-test", run_id="f" * 32,
        )
        assert "volumes:" not in yaml_str

    def test_inspect_inventory_validation_rejects_empty(self):
        failures = self.script._cfg004_validate_inspect_inventory(
            {"candidates": []},
            {"controls": [{"provider": "openai", "model": "gpt-5.3-codex"}]},
            phase_label="test",
        )
        assert any("vacuous" in f for f in failures)

    def test_inspect_inventory_validation_rejects_wrong_target_count(self):
        failures = self.script._cfg004_validate_inspect_inventory(
            {"candidates": [
                {"provider": "openai", "model": "gpt-5.3-codex", "role": "control"},
            ]},
            {"controls": [{"provider": "openai", "model": "gpt-5.3-codex"}]},
            phase_label="test",
        )
        assert any("exactly 1 qwen3.6 target" in f for f in failures)


class TestCfg004ProofTranscriptClassification:
    """Proof residual failures that are session-history collaboration/tool
    evidence gaps are suppressed when transcript validation directly covers
    them with zero failures. Any unrecognized residual stays fail-closed."""

    @pytest.fixture(autouse=True)
    def _load_script(self):
        self.script = _load_acceptance_script()

    _PROOF_CASE = "native_openai_passthrough_responses_codex_basic_alias_collaboration"

    def _saved_collab_tool_failures(self) -> list[str]:
        # Exact saved pass-3 redacted proof failures (family prefix included).
        p = f"{self._PROOF_CASE}__cfg004_clear_proof"
        return [
            f"{p} Codex command executions completed unexpected commands: expected ['git rev-parse --show-toplevel', 'git status --short', 'pwd'], got []",
            f"{p} Codex command executions did not overlap as one parallel batch: expected >= 3, got 0",
            f"{p} Codex command executions started unexpected commands: expected ['git rev-parse --show-toplevel', 'git status --short', 'pwd'], got []",
            f"{p} Codex command executions were not recorded in one turn: []",
            f"{p} missing completed Codex collaboration calls for 'spawn_agent': expected >= 1, got 0",
            f"{p} successful Codex wait did not record agents_states",
        ]

    def _run_live(self, *, proof_failures: list[str], transcript_failures: list[str]) -> dict:
        script = self.script
        cases = {
            self._PROOF_CASE: {
                "verification_alias": "basic",
                "exact_child_prompt": "child prompt",
            }
        }
        prepare_resp = {
            "result": "prepared",
            "target": {"provider": "alibaba_token_plan", "model": "alibaba_token_plan/qwen3.6-flash"},
            "control_count": 1,
            "controls": [{"provider": "openai", "model": "gpt-5.3-codex", "route_family": "codex_openai_responses"}],
        }
        inspect_active = {
            "result": "inspected",
            "candidates": [
                {"provider": "alibaba_token_plan", "model": "alibaba_token_plan/qwen3.6-flash",
                 "route_family": "codex_alibaba_token_plan_chat_completions_adapter",
                 "role": "target", "local_cooldown_active": True, "durable_cooldown_active": True},
                {"provider": "openai", "model": "gpt-5.3-codex",
                 "route_family": "codex_openai_responses",
                 "role": "control", "local_cooldown_active": True, "durable_cooldown_active": True},
            ],
        }
        inspect_target_cleared = {
            "result": "inspected",
            "candidates": [
                {"provider": "alibaba_token_plan", "model": "alibaba_token_plan/qwen3.6-flash",
                 "route_family": "codex_alibaba_token_plan_chat_completions_adapter",
                 "role": "target", "local_cooldown_active": False, "durable_cooldown_active": False},
                {"provider": "openai", "model": "gpt-5.3-codex",
                 "route_family": "codex_openai_responses",
                 "role": "control", "local_cooldown_active": True, "durable_cooldown_active": True},
            ],
        }

        def _fake_proof_case(**kwargs):
            return {
                "result": {
                    "passed": False,
                    "failures": proof_failures,
                    "langfuse": {"command_thread_id": "test-thread-id"},
                },
                "selection": {
                    "provider": "alibaba_token_plan",
                    "model": "alibaba_token_plan/qwen3.6-flash",
                    "route_family": "codex_alibaba_token_plan_chat_completions_adapter",
                },
                "phase_session_id": "test-session",
                "phase_start_time": "2026-01-01T00:00:00Z",
            }

        with (
            patch.object(script, "_cfg004_docker_compose_up", return_value=(True, "")),
            patch.object(script, "_cfg004_wait_readiness", return_value=(True, "")),
            patch.object(script, "_cfg004_build_disposable_config", return_value="/tmp/x.yaml"),
            patch.object(script, "_cfg004_remove_disposable_config"),
            patch.object(script, "_cfg004_readiness_snapshot", return_value={
                "status": 200, "config_hash": "abc", "config_version": "abc", "state": "active",
            }),
            patch.object(script, "_cfg004_docker_inspect_container", return_value={
                "id": "x", "image": "y", "started_at": "z", "restart_count": 0,
            }),
            patch.object(script, "_cfg004_docker_inspect_env_absent", return_value=[]),
            patch.object(script, "_cfg004_post_acceptance", side_effect=[
                (200, prepare_resp),
                (200, inspect_active),
                (200, inspect_target_cleared),
                (200, inspect_target_cleared),
                (200, {"result": "restored", "cleared_identities": 2}),
            ]),
            patch.object(script, "_cfg004_post_clear", return_value=(
                200, {"result": "cleared", "keys_cleared": 1, "members_removed": 1}
            )),
            patch.object(script, "_cfg003_run_proof_case", side_effect=_fake_proof_case),
            patch.object(
                script,
                "_cfg004_validate_transcript_collaboration",
                return_value=(
                    {"enabled": True, "thread_id": "test-thread-id", "total_child_commands": 3},
                    transcript_failures,
                ),
            ),
        ):
            return script._cfg004_cooldown_clear_live_test(
                litellm_base_url="http://127.0.0.1:4001",
                cases=cases,
                suite_config={},
                query_url="http://localhost:3000",
                public_key="pk-test",
                secret_key="sk-test",
            )

    def test_saved_collab_tool_proof_gaps_accepted_with_clean_transcript(self):
        result = self._run_live(
            proof_failures=self._saved_collab_tool_failures(),
            transcript_failures=[],
        )
        # Saved proof shape remains redacted-failed, but the transcript
        # replay directly covers every residual gap.
        assert result["phases"]["proof"]["passed"] is False
        assert not any(
            "did not pass endpoint/shape/session/tool validations" in f
            for f in result["failures"]
        )
        assert result["passed"] is True

    def test_unrecognized_residual_proof_failure_stays_fail_closed(self):
        result = self._run_live(
            proof_failures=(
                self._saved_collab_tool_failures()
                + [
                    f"{self._PROOF_CASE}__cfg004_clear_proof "
                    "endpoint request/response shape mismatch"
                ]
            ),
            transcript_failures=[],
        )
        assert any(
            "did not pass endpoint/shape/session/tool validations" in f
            for f in result["failures"]
        )
        assert result["passed"] is False


# ---------------------------------------------------------------------------
# Integration: prepare + real handle_cooldown_clear
# ---------------------------------------------------------------------------

_CLEAR_MOD = "litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear"


class TestPrepareThenRealClear:
    """Seed via _handle_prepare, then invoke the real
    cooldown_clear.handle_cooldown_clear, proving target cleared and
    controls remain cooled."""

    @pytest.mark.asyncio
    @patch(f"{_MOD}.rollback_cooldown_transaction", new_callable=AsyncMock)
    @patch(f"{_MOD}.publish_cooldown_transaction", new_callable=AsyncMock)
    @patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
    @patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
    @patch("litellm.proxy.proxy_server", new_callable=_mock_proxy_server, create=True)
    async def test_real_clear_handler_clears_target_controls_remain(
        self, _ps, _snap, mock_dc, mock_inspect, mock_publish, mock_rollback
    ):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
            handle_cooldown_clear,
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
            IdentitySetInspection,
        )

        mock_dc.return_value = _mock_dual_cache()
        # prepare: prestate absent (2), post-seed present (2)
        mock_inspect.side_effect = [
            _mock_inspection(exists=False, cardinality=0),
            _mock_inspection(exists=False, cardinality=0),
            _mock_inspection(exists=True, cardinality=1),
            _mock_inspection(exists=True, cardinality=1),
        ]
        mock_publish.return_value = _mock_txn_result()

        state_mgr = AliasRoutingStateManager()
        request = _make_request()
        body = {
            "operation": "prepare", "run_id": _RUN_ID, "alias": "basic",
            "ingress": "codex", "provider": "alibaba_token_plan",
            "model": "alibaba_token_plan/qwen3.6-flash", "ttl_seconds": 300,
        }
        result = await _handle_prepare(body, _RUN_ID, request, state_mgr)
        assert result["result"] == "prepared"

        # Verify both target and control are cooled locally.
        family_state = state_mgr.codex
        now = time.monotonic()
        prepared = _prepared_runs[_RUN_ID]
        for lk in prepared.all_lane_keys:
            assert family_state.cooldown_until_monotonic_by_key.get(lk, 0.0) > now

        # Now invoke the REAL clear handler targeting only the Alibaba model.
        clear_request = MagicMock()
        clear_request.json = AsyncMock(return_value={
            "provider": "alibaba_token_plan",
            "model": "alibaba_token_plan/qwen3.6-flash",
            "ingress": "codex",
        })
        admin = _make_admin_key_dict(api_key=_MASTER_KEY)

        active_insp = IdentitySetInspection(
            identity_key="idkey", exists=True,
            members=frozenset(), cardinality=1, ttl_remaining_seconds=300.0,
        )
        absent_insp = IdentitySetInspection(
            identity_key="idkey", exists=False,
            members=frozenset(), cardinality=0, ttl_remaining_seconds=None,
        )

        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
            ClearTransactionJournal,
            ClearTransactionResult,
        )
        clear_txn = ClearTransactionResult(
            transaction_id="clear-txn",
            phase="CLEAR_COMMITTED",
            journal=ClearTransactionJournal(
                transaction_id="clear-txn", phase="CLEAR_COMMITTED",
                alias_family="codex", identity_hash="ih",
                cooldown_keys=["k"], lane_members=["k"],
                expected_members=["k"], identity_key="idkey",
                receipt_key="rkey", receipt_ttl=300,
            ),
            keys_deleted=1, members_removed=1,
        )

        with (
            patch.dict("os.environ", {"AAWM_ALIAS_ROUTING_COOLDOWN_CLEAR_SINGLE_WORKER": "1"}),
            patch(f"{_CLEAR_MOD}.get_active_routing_snapshot", return_value=_make_snapshot()),
            patch("litellm.proxy.proxy_server.master_key", _MASTER_KEY),
            patch(f"{_CLEAR_MOD}.alias_routing_state", state_mgr),
            patch(f"{_CLEAR_MOD}.get_aawm_alias_routing_dual_cache", return_value=_mock_dual_cache()),
            patch(f"{_CLEAR_MOD}.inspect_identity_set", new_callable=AsyncMock,
                  side_effect=[active_insp, active_insp, active_insp, active_insp, absent_insp]),
            patch(f"{_CLEAR_MOD}.clear_cooldown_transaction", new_callable=AsyncMock,
                  return_value=clear_txn),
            patch(f"{_CLEAR_MOD}.verify_aawm_alias_routing_durable_absence",
                  new_callable=AsyncMock, return_value=True),
            patch(f"{_CLEAR_MOD}._emit_audit_event"),
        ):
            clear_result = await handle_cooldown_clear(clear_request, admin)

        assert clear_result["result"] == "cleared"
        assert clear_result["keys_cleared"] == 1
        assert clear_result["members_removed"] == 1

        # Target lane cleared, control lane still cooled.
        now = time.monotonic()
        assert family_state.cooldown_until_monotonic_by_key.get(
            prepared.target_lane_key, 0.0
        ) <= now, "target lane should be cleared"
        for clk in prepared.control_lane_keys:
            assert family_state.cooldown_until_monotonic_by_key.get(
                clk, 0.0
            ) > now, f"control lane {clk} should remain cooled"


# ---------------------------------------------------------------------------
# Defect 4: rejection tests
# ---------------------------------------------------------------------------


class TestInventoryRejections:
    """Focused rejection tests for empty controls, route mismatch,
    non-200 post-TUI, and count mismatch."""

    @pytest.fixture(autouse=True)
    def _load_script(self):
        self.script = _load_acceptance_script()

    def test_empty_controls_rejected(self):
        """Prepare must reject when no control candidates exist."""
        failures = self.script._cfg004_validate_inspect_inventory(
            {"candidates": [
                {"provider": "alibaba_token_plan", "model": "alibaba_token_plan/qwen3.6-flash",
                 "role": "target"},
            ]},
            {"controls": []},
            phase_label="test",
        )
        assert any("no controls" in f for f in failures)

    def test_route_family_mismatch_rejected(self):
        """Control identity includes route_family; mismatch must fail."""
        failures = self.script._cfg004_validate_inspect_inventory(
            {"candidates": [
                {"provider": "alibaba_token_plan", "model": "alibaba_token_plan/qwen3.6-flash",
                 "role": "target"},
                {"provider": "openai", "model": "gpt-5.3-codex",
                 "route_family": "WRONG_FAMILY", "role": "control"},
            ]},
            {"controls": [
                {"provider": "openai", "model": "gpt-5.3-codex",
                 "route_family": "codex_openai_responses"},
            ]},
            phase_label="test",
        )
        assert any("control inventory mismatch" in f for f in failures)

    def test_non_200_post_tui_rejected(self):
        """Post-TUI inspect returning non-200 must fail the live test."""
        proof_case = "native_openai_passthrough_responses_codex_basic_alias_collaboration"
        cases = {proof_case: {"verification_alias": "basic"}}
        prepare_resp = {
            "result": "prepared",
            "target": {"provider": "alibaba_token_plan", "model": "alibaba_token_plan/qwen3.6-flash"},
            "control_count": 1,
            "controls": [{"provider": "openai", "model": "gpt-5.3-codex", "route_family": "codex_openai_responses"}],
        }
        inspect_ok = {
            "result": "inspected",
            "candidates": [
                {"provider": "alibaba_token_plan", "model": "alibaba_token_plan/qwen3.6-flash",
                 "role": "target", "local_cooldown_active": True, "durable_cooldown_active": True},
                {"provider": "openai", "model": "gpt-5.3-codex", "route_family": "codex_openai_responses",
                 "role": "control", "local_cooldown_active": True, "durable_cooldown_active": True},
            ],
        }
        inspect_target_cleared = {
            "result": "inspected",
            "candidates": [
                {"provider": "alibaba_token_plan", "model": "alibaba_token_plan/qwen3.6-flash",
                 "role": "target", "local_cooldown_active": False, "durable_cooldown_active": False},
                {"provider": "openai", "model": "gpt-5.3-codex", "route_family": "codex_openai_responses",
                 "role": "control", "local_cooldown_active": True, "durable_cooldown_active": True},
            ],
        }

        def _fake_proof(**kw):
            return {
                "result": {"passed": True},
                "selection": {"provider": "alibaba_token_plan",
                              "model": "alibaba_token_plan/qwen3.6-flash",
                              "route_family": "codex_alibaba_token_plan_chat_completions_adapter"},
                "phase_session_id": "s", "phase_start_time": "t",
            }

        with (
            patch.object(self.script, "_cfg004_docker_compose_up", return_value=(True, "")),
            patch.object(self.script, "_cfg004_wait_readiness", return_value=(True, "")),
            patch.object(self.script, "_cfg004_readiness_snapshot", return_value={
                "status": 200, "config_hash": "abc", "config_version": "abc", "state": "active",
            }),
            patch.object(self.script, "_cfg004_docker_inspect_container", return_value={
                "id": "x", "image": "y", "started_at": "z", "restart_count": 0,
            }),
            patch.object(self.script, "_cfg004_docker_inspect_env_absent", return_value=[]),
            patch.object(self.script, "_cfg004_post_acceptance", side_effect=[
                (200, prepare_resp),
                (200, inspect_ok),       # pre-clear
                (200, inspect_target_cleared),  # post-clear
                (503, {}),               # post-TUI: non-200
                (200, {"result": "restored", "cleared_identities": 2}),
            ]),
            patch.object(self.script, "_cfg004_post_clear", return_value=(
                200, {"result": "cleared", "keys_cleared": 1, "members_removed": 1}
            )),
            patch.object(self.script, "_cfg003_run_proof_case", side_effect=_fake_proof),
            patch.object(self.script, "RA") as mock_ra,
        ):
            mock_ra._redact_sensitive_artifact_fields = lambda x: x
            mock_ra._HEALTH_READINESS_PATH = "/health/readiness"
            result = self.script._cfg004_cooldown_clear_live_test(
                litellm_base_url="http://127.0.0.1:4001",
                cases=cases, suite_config={},
                query_url="http://localhost:3000",
                public_key="pk", secret_key="sk",
            )
        assert result["passed"] is False
        assert any("post-proof inspect failed" in f for f in result["failures"])

    def test_count_mismatch_rejected(self):
        """keys_cleared != 1 must fail."""
        proof_case = "native_openai_passthrough_responses_codex_basic_alias_collaboration"
        cases = {proof_case: {"verification_alias": "basic"}}
        prepare_resp = {
            "result": "prepared",
            "target": {"provider": "alibaba_token_plan", "model": "alibaba_token_plan/qwen3.6-flash"},
            "control_count": 1,
            "controls": [{"provider": "openai", "model": "gpt-5.3-codex", "route_family": "codex_openai_responses"}],
        }
        inspect_ok = {
            "result": "inspected",
            "candidates": [
                {"provider": "alibaba_token_plan", "model": "alibaba_token_plan/qwen3.6-flash",
                 "role": "target", "local_cooldown_active": True, "durable_cooldown_active": True},
                {"provider": "openai", "model": "gpt-5.3-codex", "route_family": "codex_openai_responses",
                 "role": "control", "local_cooldown_active": True, "durable_cooldown_active": True},
            ],
        }

        with (
            patch.object(self.script, "_cfg004_docker_compose_up", return_value=(True, "")),
            patch.object(self.script, "_cfg004_wait_readiness", return_value=(True, "")),
            patch.object(self.script, "_cfg004_readiness_snapshot", return_value={
                "status": 200, "config_hash": "abc", "config_version": "abc", "state": "active",
            }),
            patch.object(self.script, "_cfg004_docker_inspect_container", return_value={
                "id": "x", "image": "y", "started_at": "z", "restart_count": 0,
            }),
            patch.object(self.script, "_cfg004_post_acceptance", side_effect=[
                (200, prepare_resp),
                (200, inspect_ok),
                (200, inspect_ok),  # post-clear
                (200, inspect_ok),  # post-proof
                (200, {"result": "restored", "cleared_identities": 2}),
            ]),
            patch.object(self.script, "_cfg004_post_clear", return_value=(
                200, {"result": "cleared", "keys_cleared": 2, "members_removed": 1}
            )),
            patch.object(self.script, "_cfg003_run_proof_case", side_effect=lambda **kw: {
                "result": {"passed": True},
                "selection": {"provider": "alibaba_token_plan",
                              "model": "alibaba_token_plan/qwen3.6-flash",
                              "route_family": "codex_alibaba_token_plan_chat_completions_adapter"},
                "phase_session_id": "s", "phase_start_time": "t",
            }),
            patch.object(self.script, "_cfg004_docker_inspect_env_absent", return_value=[]),
            patch.object(self.script, "RA") as mock_ra,
        ):
            mock_ra._redact_sensitive_artifact_fields = lambda x: x
            mock_ra._HEALTH_READINESS_PATH = "/health/readiness"
            result = self.script._cfg004_cooldown_clear_live_test(
                litellm_base_url="http://127.0.0.1:4001",
                cases=cases, suite_config={},
                query_url="http://localhost:3000",
                public_key="pk", secret_key="sk",
            )
        assert result["passed"] is False
        assert any("keys_cleared must be exactly 1" in f for f in result["failures"])


# ---------------------------------------------------------------------------
# CFG-004 criterion 11 initiation-2: auth bypass tests
# ---------------------------------------------------------------------------

_PASSTHROUGH_MOD = "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints"


def _make_bypass_request(path: str = "/openai_passthrough/v1/responses") -> MagicMock:
    """Build a mock Request with url.path for bypass predicate."""
    req = MagicMock()
    req.url = MagicMock()
    req.url.path = path
    return req


def _bypass_env(run_id: str = _RUN_ID) -> dict[str, str]:
    return {
        "AAWM_LITELLM_ENVIRONMENT": "litellm-dev",
        "AAWM_CFG004_ACCEPTANCE_ENABLED": "1",
        "AAWM_CFG004_ACCEPTANCE_RUN_ID": run_id,
        "AAWM_ALIAS_ROUTING_COOLDOWN_CLEAR_SINGLE_WORKER": "1",
    }


class TestCfg004AuthBypass:
    """Focused tests for the /openai_passthrough acceptance auth bypass."""

    @patch(f"{_PASSTHROUGH_MOD}.user_api_key_auth", new_callable=AsyncMock)
    @patch.dict("os.environ", _bypass_env(), clear=False)
    @patch("litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable.get_aawm_alias_routing_state_namespace",
           return_value=f"aawm-routing-dev-cfg004-{_RUN_ID}", create=True)
    @patch("litellm.proxy.proxy_server.general_settings",
           {"public_routes": ["/openai_passthrough/*"]}, create=True)
    async def test_bypass_succeeds_under_complete_gates(self, _mock_ns, mock_auth):
        """Auth bypass returns INTERNAL_USER_VIEW_ONLY without calling
        user_api_key_auth when all gates pass."""
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _cfg004_openai_passthrough_auth_dependency,
        )

        request = _make_bypass_request()
        result = await _cfg004_openai_passthrough_auth_dependency(
            request=request, endpoint="v1/responses", api_key="Bearer oauth-token"
        )
        from litellm.proxy._types import LitellmUserRoles
        assert result.user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
        mock_auth.assert_not_called()

    @patch(f"{_PASSTHROUGH_MOD}.user_api_key_auth", new_callable=AsyncMock)
    @patch.dict("os.environ", {**_bypass_env(), "AAWM_LITELLM_ENVIRONMENT": "prod"}, clear=False)
    async def test_bypass_denied_wrong_environment(self, mock_auth):
        """Wrong environment delegates to user_api_key_auth."""
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _cfg004_openai_passthrough_auth_dependency,
        )

        mock_auth.return_value = MagicMock()
        request = _make_bypass_request()
        await _cfg004_openai_passthrough_auth_dependency(
            request=request, endpoint="v1/responses", api_key="sk-test"
        )
        mock_auth.assert_called_once()

    @patch(f"{_PASSTHROUGH_MOD}.user_api_key_auth", new_callable=AsyncMock)
    @patch.dict("os.environ", {**_bypass_env(), "AAWM_CFG004_ACCEPTANCE_ENABLED": "0"}, clear=False)
    async def test_bypass_denied_not_enabled(self, mock_auth):
        """Disabled flag delegates to user_api_key_auth."""
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _cfg004_openai_passthrough_auth_dependency,
        )

        mock_auth.return_value = MagicMock()
        request = _make_bypass_request()
        await _cfg004_openai_passthrough_auth_dependency(
            request=request, endpoint="v1/responses", api_key="sk-test"
        )
        mock_auth.assert_called_once()

    @patch(f"{_PASSTHROUGH_MOD}.user_api_key_auth", new_callable=AsyncMock)
    @patch.dict("os.environ", {**_bypass_env(), "AAWM_CFG004_ACCEPTANCE_RUN_ID": "not-hex"}, clear=False)
    async def test_bypass_denied_invalid_run_id(self, mock_auth):
        """Invalid run_id delegates to user_api_key_auth."""
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _cfg004_openai_passthrough_auth_dependency,
        )

        mock_auth.return_value = MagicMock()
        request = _make_bypass_request()
        await _cfg004_openai_passthrough_auth_dependency(
            request=request, endpoint="v1/responses", api_key="sk-test"
        )
        mock_auth.assert_called_once()

    @patch(f"{_PASSTHROUGH_MOD}.user_api_key_auth", new_callable=AsyncMock)
    @patch.dict("os.environ", _bypass_env(), clear=False)
    @patch("litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable.get_aawm_alias_routing_state_namespace",
           return_value="wrong-namespace", create=True)
    async def test_bypass_denied_namespace_mismatch(self, _mock_ns, mock_auth):
        """Namespace mismatch delegates to user_api_key_auth."""
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _cfg004_openai_passthrough_auth_dependency,
        )

        mock_auth.return_value = MagicMock()
        request = _make_bypass_request()
        await _cfg004_openai_passthrough_auth_dependency(
            request=request, endpoint="v1/responses", api_key="sk-test"
        )
        mock_auth.assert_called_once()

    @patch(f"{_PASSTHROUGH_MOD}.user_api_key_auth", new_callable=AsyncMock)
    @patch.dict("os.environ", _bypass_env(), clear=False)
    @patch("litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable.get_aawm_alias_routing_state_namespace",
           return_value=f"aawm-routing-dev-cfg004-{_RUN_ID}", create=True)
    @patch("litellm.proxy.proxy_server.general_settings",
           {"public_routes": ["/other/*"]}, create=True)
    async def test_bypass_denied_wrong_public_routes(self, _mock_ns, mock_auth):
        """Missing exact public-route declaration delegates to auth."""
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _cfg004_openai_passthrough_auth_dependency,
        )

        mock_auth.return_value = MagicMock()
        request = _make_bypass_request()
        await _cfg004_openai_passthrough_auth_dependency(
            request=request, endpoint="v1/responses", api_key="sk-test"
        )
        mock_auth.assert_called_once()

    @patch(f"{_PASSTHROUGH_MOD}.user_api_key_auth", new_callable=AsyncMock)
    @patch.dict("os.environ", _bypass_env(), clear=False)
    @patch("litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable.get_aawm_alias_routing_state_namespace",
           return_value=f"aawm-routing-dev-cfg004-{_RUN_ID}", create=True)
    @patch("litellm.proxy.proxy_server.general_settings",
           {"public_routes": ["/openai_passthrough/*"]}, create=True)
    async def test_bypass_denied_non_responses_endpoint(self, _mock_ns, mock_auth):
        """Non-Responses endpoint delegates to user_api_key_auth."""
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _cfg004_openai_passthrough_auth_dependency,
        )

        mock_auth.return_value = MagicMock()
        request = _make_bypass_request()
        await _cfg004_openai_passthrough_auth_dependency(
            request=request, endpoint="v1/chat/completions", api_key="sk-test"
        )
        mock_auth.assert_called_once()

    @patch(f"{_PASSTHROUGH_MOD}.user_api_key_auth", new_callable=AsyncMock)
    @patch.dict("os.environ", _bypass_env(), clear=False)
    @patch("litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable.get_aawm_alias_routing_state_namespace",
           return_value=f"aawm-routing-dev-cfg004-{_RUN_ID}", create=True)
    @patch("litellm.proxy.proxy_server.general_settings",
           {"public_routes": ["/openai_passthrough/*"]}, create=True)
    async def test_bypass_denied_openai_route_family(self, _mock_ns, mock_auth):
        """/openai/* path (not /openai_passthrough/) delegates to auth."""
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _cfg004_openai_passthrough_auth_dependency,
        )

        mock_auth.return_value = MagicMock()
        request = _make_bypass_request(path="/openai/v1/responses")
        await _cfg004_openai_passthrough_auth_dependency(
            request=request, endpoint="v1/responses", api_key="sk-test"
        )
        mock_auth.assert_called_once()

    @patch(f"{_PASSTHROUGH_MOD}.user_api_key_auth", new_callable=AsyncMock)
    @patch.dict("os.environ", {}, clear=False)
    async def test_responses_client_auth_is_not_forwarded_server_inventory_supplies_credentials(
        self, mock_auth
    ):
        """OPENAI-006: Responses does not preserve/forward inbound client auth.

        CFG-004 still delegates auth dependency normally when unset, but
        Responses credential egress is supplied by server Codex OAuth inventory
        rather than inbound client Authorization forwarding.
        """
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _cfg004_openai_passthrough_auth_dependency,
            _should_preserve_openai_client_auth,
        )

        mock_auth.return_value = MagicMock()
        request = _make_bypass_request()
        request.headers = {"authorization": "Bearer oauth-token"}
        await _cfg004_openai_passthrough_auth_dependency(
            request=request, endpoint="v1/responses", api_key="Bearer oauth-token"
        )
        mock_auth.assert_called_once()
        assert _should_preserve_openai_client_auth(request, "v1/responses") is False


# ---------------------------------------------------------------------------
# CFG-004 criterion 11 initiation-2: restore ordering tests
# ---------------------------------------------------------------------------


class TestRestoreOrdering:
    """Prove target receipt deletion occurs AFTER control rollback and
    postcondition checks; rollback failure retains receipt + recovery state."""

    @pytest.fixture
    def state_mgr(self):
        return AliasRoutingStateManager()

    @pytest.mark.asyncio
    @patch(f"{_MOD}.verify_aawm_alias_routing_durable_absence", new_callable=AsyncMock)
    @patch(f"{_MOD}.delete_aawm_alias_routing_durable_key", new_callable=AsyncMock)
    @patch(f"{_MOD}.rollback_cooldown_transaction", new_callable=AsyncMock)
    @patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
    @patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
    @patch("litellm.proxy.proxy_server", new_callable=_mock_proxy_server, create=True)
    async def test_receipt_deleted_after_rollback_and_verification(
        self, _ps, _snap, mock_dc, mock_inspect, mock_rollback, mock_delete, mock_verify, state_mgr
    ):
        """Target receipt deletion is the final cleanup step, after control
        rollback and postcondition verification."""
        mock_dc.return_value = _mock_dual_cache()
        mock_inspect.return_value = _mock_inspection(exists=False, cardinality=0)
        mock_verify.return_value = True
        mock_delete.return_value = True

        call_order: list[str] = []
        mock_rollback.side_effect = lambda **kw: call_order.append("rollback")
        mock_delete.side_effect = lambda **kw: call_order.append("delete")

        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_acceptance import _PreparedState
        _prepared_runs[_RUN_ID] = _PreparedState(
            run_id=_RUN_ID,
            target_identity_hash="tih",
            target_lane_key="tlk",
            control_identity_hashes=["cih"],
            control_lane_keys=["clk"],
            all_identity_hashes=["tih", "cih"],
            all_lane_keys=["tlk", "clk"],
            publication_results=[_mock_txn_result()],
            target_publication_result=_mock_txn_result(),
            control_publication_results=[_mock_txn_result()],
            local_preimages={},
            prepared_at=time.monotonic(),
        )

        result = await _handle_restore(_RUN_ID, state_mgr)
        assert result["result"] == "restored"
        assert _RUN_ID not in _prepared_runs
        # Rollback must precede delete.
        assert call_order.index("rollback") < call_order.index("delete")

    @pytest.mark.asyncio
    @patch(f"{_MOD}.verify_aawm_alias_routing_durable_absence", new_callable=AsyncMock)
    @patch(f"{_MOD}.delete_aawm_alias_routing_durable_key", new_callable=AsyncMock)
    @patch(f"{_MOD}.rollback_cooldown_transaction", new_callable=AsyncMock)
    @patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
    @patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
    @patch("litellm.proxy.proxy_server", new_callable=_mock_proxy_server, create=True)
    async def test_rollback_failure_retains_receipt_and_recovery_state(
        self, _ps, _snap, mock_dc, mock_inspect, mock_rollback, mock_delete, mock_verify, state_mgr
    ):
        """When control rollback raises, target receipt is NOT deleted and
        _prepared_runs is retained with structured error detail."""
        mock_dc.return_value = _mock_dual_cache()
        mock_inspect.return_value = _mock_inspection(exists=False, cardinality=0)
        mock_rollback.side_effect = RuntimeError("redis down")

        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_acceptance import _PreparedState
        _prepared_runs[_RUN_ID] = _PreparedState(
            run_id=_RUN_ID,
            target_identity_hash="tih",
            target_lane_key="tlk",
            control_identity_hashes=["cih"],
            control_lane_keys=["clk"],
            all_identity_hashes=["tih", "cih"],
            all_lane_keys=["tlk", "clk"],
            publication_results=[_mock_txn_result()],
            target_publication_result=_mock_txn_result(),
            control_publication_results=[_mock_txn_result()],
            local_preimages={},
            prepared_at=time.monotonic(),
        )

        with pytest.raises(HTTPException) as exc_info:
            await _handle_restore(_RUN_ID, state_mgr)
        assert exc_info.value.status_code == 500
        detail = exc_info.value.detail
        assert detail["error"] == "control_rollback_failed"
        assert detail["phase"] == "control_rollback"
        assert detail["failure_class"] == "RuntimeError"
        assert detail["run_id"] == _RUN_ID
        assert detail["recovery_state_retained"] is True
        assert detail["target_receipt_retained"] is True
        # Receipt NOT deleted.
        mock_delete.assert_not_called()
        # Recovery state retained.
        assert _RUN_ID in _prepared_runs

    @pytest.mark.asyncio
    @patch(f"{_MOD}.verify_aawm_alias_routing_durable_absence", new_callable=AsyncMock)
    @patch(f"{_MOD}.delete_aawm_alias_routing_durable_key", new_callable=AsyncMock)
    @patch(f"{_MOD}.rollback_cooldown_transaction", new_callable=AsyncMock)
    @patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
    @patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
    @patch("litellm.proxy.proxy_server", new_callable=_mock_proxy_server, create=True)
    async def test_successful_restore_cleans_and_pops(
        self, _ps, _snap, mock_dc, mock_inspect, mock_rollback, mock_delete, mock_verify, state_mgr
    ):
        """Successful restore deletes target receipt, verifies absence, and
        pops _prepared_runs."""
        mock_dc.return_value = _mock_dual_cache()
        mock_inspect.return_value = _mock_inspection(exists=False, cardinality=0)
        mock_verify.return_value = True
        mock_delete.return_value = True

        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_acceptance import _PreparedState
        _prepared_runs[_RUN_ID] = _PreparedState(
            run_id=_RUN_ID,
            target_identity_hash="tih",
            target_lane_key="tlk",
            control_identity_hashes=["cih"],
            control_lane_keys=["clk"],
            all_identity_hashes=["tih", "cih"],
            all_lane_keys=["tlk", "clk"],
            publication_results=[_mock_txn_result()],
            target_publication_result=_mock_txn_result(),
            control_publication_results=[_mock_txn_result()],
            local_preimages={},
            prepared_at=time.monotonic(),
        )

        result = await _handle_restore(_RUN_ID, state_mgr)
        assert result["result"] == "restored"
        assert result["cleared_identities"] == 2
        assert _RUN_ID not in _prepared_runs
        mock_delete.assert_called_once()
        mock_rollback.assert_called_once()


# ---------------------------------------------------------------------------
# CFG-004 independent-validation remediation 1: topology gate in auth bypass
# ---------------------------------------------------------------------------


class TestCfg004AuthBypassTopologyGate:
    """Prove _cfg004_acceptance_auth_bypass_active requires
    AAWM_ALIAS_ROUTING_COOLDOWN_CLEAR_SINGLE_WORKER=1."""

    @patch(f"{_PASSTHROUGH_MOD}.user_api_key_auth", new_callable=AsyncMock)
    @patch.dict("os.environ", {**_bypass_env(), "AAWM_ALIAS_ROUTING_COOLDOWN_CLEAR_SINGLE_WORKER": "0"}, clear=False)
    @patch("litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable.get_aawm_alias_routing_state_namespace",
           return_value=f"aawm-routing-dev-cfg004-{_RUN_ID}", create=True)
    @patch("litellm.proxy.proxy_server.general_settings",
           {"public_routes": ["/openai_passthrough/*"]}, create=True)
    async def test_bypass_denied_topology_zero(self, _mock_ns, mock_auth):
        """Topology=0 delegates to user_api_key_auth."""
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _cfg004_openai_passthrough_auth_dependency,
        )

        mock_auth.return_value = MagicMock()
        request = _make_bypass_request()
        await _cfg004_openai_passthrough_auth_dependency(
            request=request, endpoint="v1/responses", api_key="sk-test"
        )
        mock_auth.assert_called_once()

    @patch(f"{_PASSTHROUGH_MOD}.user_api_key_auth", new_callable=AsyncMock)
    @patch.dict("os.environ", {k: v for k, v in _bypass_env().items()
                               if k != "AAWM_ALIAS_ROUTING_COOLDOWN_CLEAR_SINGLE_WORKER"}, clear=False)
    @patch("litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable.get_aawm_alias_routing_state_namespace",
           return_value=f"aawm-routing-dev-cfg004-{_RUN_ID}", create=True)
    @patch("litellm.proxy.proxy_server.general_settings",
           {"public_routes": ["/openai_passthrough/*"]}, create=True)
    async def test_bypass_denied_topology_unset(self, _mock_ns, mock_auth):
        """Missing topology var delegates to user_api_key_auth."""
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _cfg004_acceptance_auth_bypass_active,
        )

        request = _make_bypass_request()
        assert _cfg004_acceptance_auth_bypass_active(request, "v1/responses") is False

    @patch(f"{_PASSTHROUGH_MOD}.user_api_key_auth", new_callable=AsyncMock)
    @patch.dict("os.environ", _bypass_env(), clear=False)
    @patch("litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable.get_aawm_alias_routing_state_namespace",
           return_value=f"aawm-routing-dev-cfg004-{_RUN_ID}", create=True)
    @patch("litellm.proxy.proxy_server.general_settings",
           {"public_routes": ["/openai_passthrough/*"]}, create=True)
    async def test_bypass_positive_with_topology(self, _mock_ns, mock_auth):
        """Complete gates including topology=1 bypass auth."""
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _cfg004_acceptance_auth_bypass_active,
        )

        request = _make_bypass_request()
        assert _cfg004_acceptance_auth_bypass_active(request, "v1/responses") is True


# ---------------------------------------------------------------------------
# CFG-004 independent-validation remediation 2: non-vacuous target absence
# ---------------------------------------------------------------------------


class TestRestoreTargetAbsenceNonVacuous:
    """Prove target local absence is checked BEFORE restore clears local
    state, so an active local target produces target_absence_not_proven."""

    @pytest.fixture
    def state_mgr(self):
        return AliasRoutingStateManager()

    @pytest.mark.asyncio
    @patch(f"{_MOD}.verify_aawm_alias_routing_durable_absence", new_callable=AsyncMock)
    @patch(f"{_MOD}.delete_aawm_alias_routing_durable_key", new_callable=AsyncMock)
    @patch(f"{_MOD}.rollback_cooldown_transaction", new_callable=AsyncMock)
    @patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
    @patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
    @patch("litellm.proxy.proxy_server", new_callable=_mock_proxy_server, create=True)
    async def test_active_local_target_blocks_restore(
        self, _ps, _snap, mock_dc, mock_inspect, mock_rollback, mock_delete, mock_verify, state_mgr
    ):
        """Seeding an active local target cooldown must produce structured
        target_absence_not_proven with _prepared_runs retained."""
        mock_dc.return_value = _mock_dual_cache()
        # Durable target absent, but local is active.
        mock_inspect.return_value = _mock_inspection(exists=False, cardinality=0)

        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_acceptance import _PreparedState
        _prepared_runs[_RUN_ID] = _PreparedState(
            run_id=_RUN_ID,
            target_identity_hash="tih",
            target_lane_key="tlk",
            control_identity_hashes=["cih"],
            control_lane_keys=["clk"],
            all_identity_hashes=["tih", "cih"],
            all_lane_keys=["tlk", "clk"],
            publication_results=[_mock_txn_result()],
            target_publication_result=_mock_txn_result(),
            control_publication_results=[_mock_txn_result()],
            local_preimages={},
            prepared_at=time.monotonic(),
        )

        # Seed an ACTIVE local cooldown on the target lane key.
        family_state = state_mgr.codex
        async with family_state.lock:
            family_state.cooldown_until_monotonic_by_key["tlk"] = time.monotonic() + 9999

        with pytest.raises(HTTPException) as exc_info:
            await _handle_restore(_RUN_ID, state_mgr)

        detail = exc_info.value.detail
        assert detail["error"] == "target_absence_not_proven"
        assert detail["phase"] == "target_absence_proof"
        assert detail["failure_class"] == "precondition"
        assert detail["run_id"] == _RUN_ID
        assert detail["recovery_state_retained"] is True
        assert detail["target_receipt_retained"] is True
        assert "target local cooldown still active" in detail["failures"][0]
        # _prepared_runs retained.
        assert _RUN_ID in _prepared_runs
        # No rollback or delete attempted.
        mock_rollback.assert_not_called()
        mock_delete.assert_not_called()


# ---------------------------------------------------------------------------
# CFG-004 independent-validation remediation 3: exception sanitization
# ---------------------------------------------------------------------------


class TestRestoreExceptionSanitization:
    """Prove RuntimeError('Bearer secret-sentinel') never escapes into
    HTTPException detail from any restore phase."""

    @pytest.fixture
    def state_mgr(self):
        return AliasRoutingStateManager()

    def _seed_prepared(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_acceptance import _PreparedState
        _prepared_runs[_RUN_ID] = _PreparedState(
            run_id=_RUN_ID,
            target_identity_hash="tih",
            target_lane_key="tlk",
            control_identity_hashes=["cih"],
            control_lane_keys=["clk"],
            all_identity_hashes=["tih", "cih"],
            all_lane_keys=["tlk", "clk"],
            publication_results=[_mock_txn_result()],
            target_publication_result=_mock_txn_result(),
            control_publication_results=[_mock_txn_result()],
            local_preimages={},
            prepared_at=time.monotonic(),
        )

    @pytest.mark.asyncio
    @patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
    @patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
    @patch("litellm.proxy.proxy_server", new_callable=_mock_proxy_server, create=True)
    async def test_target_inspect_exception_sanitized(
        self, _ps, _snap, mock_dc, mock_inspect, state_mgr
    ):
        """inspect_identity_set raising RuntimeError('Bearer secret-sentinel')
        during target absence proof produces sanitized detail."""
        mock_dc.return_value = _mock_dual_cache()
        mock_inspect.side_effect = RuntimeError("Bearer secret-sentinel")
        self._seed_prepared()

        with pytest.raises(HTTPException) as exc_info:
            await _handle_restore(_RUN_ID, state_mgr)

        detail = exc_info.value.detail
        assert detail["error"] == "target_inspect_failed"
        assert detail["phase"] == "target_absence_proof"
        assert detail["failure_class"] == "RuntimeError"
        assert detail["run_id"] == _RUN_ID
        assert detail["recovery_state_retained"] is True
        assert detail["target_receipt_retained"] is True
        detail_str = str(detail)
        assert "secret-sentinel" not in detail_str
        assert "Bearer" not in detail_str
        assert _RUN_ID in _prepared_runs

    @pytest.mark.asyncio
    @patch(f"{_MOD}.verify_aawm_alias_routing_durable_absence", new_callable=AsyncMock)
    @patch(f"{_MOD}.delete_aawm_alias_routing_durable_key", new_callable=AsyncMock)
    @patch(f"{_MOD}.rollback_cooldown_transaction", new_callable=AsyncMock)
    @patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
    @patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
    @patch("litellm.proxy.proxy_server", new_callable=_mock_proxy_server, create=True)
    async def test_postcondition_verify_exception_sanitized(
        self, _ps, _snap, mock_dc, mock_inspect, mock_rollback, mock_delete, mock_verify, state_mgr
    ):
        """verify_aawm_alias_routing_durable_absence raising RuntimeError
        during postcondition produces sanitized detail."""
        mock_dc.return_value = _mock_dual_cache()
        mock_inspect.return_value = _mock_inspection(exists=False, cardinality=0)
        mock_verify.side_effect = RuntimeError("Bearer secret-sentinel")
        self._seed_prepared()

        with pytest.raises(HTTPException) as exc_info:
            await _handle_restore(_RUN_ID, state_mgr)

        detail = exc_info.value.detail
        assert detail["error"] == "postcondition_verify_failed"
        assert detail["phase"] == "postcondition_verification"
        assert detail["failure_class"] == "RuntimeError"
        detail_str = str(detail)
        assert "secret-sentinel" not in detail_str
        assert "Bearer" not in detail_str
        assert _RUN_ID in _prepared_runs


# ---------------------------------------------------------------------------
# CFG-004 initiation-3: RollbackDriftError during control_rollback fallback
# ---------------------------------------------------------------------------


class TestRestoreRollbackDriftFallback:
    """Prove that RollbackDriftError (TTL decay between prepare and restore)
    triggers direct-deletion fallback and restore still completes."""

    @pytest.fixture
    def state_mgr(self):
        return AliasRoutingStateManager()

    @staticmethod
    def _real_txn_result(identity_hash: str = "cih", lane_key: str = "clk", txn_id: str = "txn123"):
        """Build a real CooldownTransactionResult with real journal."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
            CooldownTransactionJournal,
            CooldownTransactionResult,
        )
        journal = CooldownTransactionJournal(
            transaction_id=txn_id,
            phase="DURABLE_COMMITTED",
            alias_family="codex",
            identity_hash=identity_hash,
            cooldown_keys=[lane_key],
            identity_keys=[f"aawm:cdr:c:lane_identity:{identity_hash}"],
            lane_members=[lane_key],
            preimages=[(None, -2)],
            receipt_key=f"aawm:cdr:c:txn_receipt:txn-receipt:{txn_id}",
            requested_ttl=300,
        )
        return CooldownTransactionResult(
            transaction_id=txn_id,
            phase="DURABLE_COMMITTED",
            journal=journal,
        )

    def _seed_prepared(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_acceptance import _PreparedState
        control_txn = self._real_txn_result(identity_hash="cih", lane_key="clk", txn_id="ctrltxn1")
        target_txn = self._real_txn_result(identity_hash="tih", lane_key="tlk", txn_id="tgtxtn1")
        _prepared_runs[_RUN_ID] = _PreparedState(
            run_id=_RUN_ID,
            target_identity_hash="tih",
            target_lane_key="tlk",
            control_identity_hashes=["cih"],
            control_lane_keys=["clk"],
            all_identity_hashes=["tih", "cih"],
            all_lane_keys=["tlk", "clk"],
            publication_results=[control_txn, target_txn],
            target_publication_result=target_txn,
            control_publication_results=[control_txn],
            local_preimages={},
            prepared_at=time.monotonic(),
        )

    @pytest.mark.asyncio
    @patch(f"{_MOD}.verify_aawm_alias_routing_durable_absence", new_callable=AsyncMock)
    @patch(f"{_MOD}._delete_durable_set_key", new_callable=AsyncMock)
    @patch(f"{_MOD}.delete_aawm_alias_routing_durable_key", new_callable=AsyncMock)
    @patch(f"{_MOD}.rollback_cooldown_transaction", new_callable=AsyncMock)
    @patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
    @patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
    @patch("litellm.proxy.proxy_server", new_callable=_mock_proxy_server, create=True)
    async def test_rollback_drift_error_triggers_fallback_and_completes(
        self, _ps, _snap, mock_dc, mock_inspect, mock_rollback, mock_delete, mock_set_delete, mock_verify, state_mgr
    ):
        """RollbackDriftError from TTL decay falls back to direct deletion;
        restore completes with full postcondition verification."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import RollbackDriftError

        mock_dc.return_value = _mock_dual_cache()
        mock_inspect.return_value = _mock_inspection(exists=False, cardinality=0)
        mock_verify.return_value = True
        mock_delete.return_value = True
        mock_set_delete.return_value = None
        # Simulate the observed failure: rollback raises RollbackDriftError.
        mock_rollback.side_effect = RollbackDriftError(
            phase="DURABLE_COMMITTED",
            family="codex",
            transaction_id_prefix="txn123",
            identity_prefix="ih",
            key_count=1,
            exception_classes=(),
        )
        self._seed_prepared()

        result = await _handle_restore(_RUN_ID, state_mgr)

        assert result["result"] == "restored"
        assert result["cleared_identities"] == 2
        assert _RUN_ID not in _prepared_runs
        # Fallback used direct deletion for cooldown key and control receipt
        # (2 via delete_aawm_alias_routing_durable_key), plus target receipt
        # cleanup in Phase 6 (1) = 3 total.
        assert mock_delete.call_count == 3
        delete_calls = [c.kwargs for c in mock_delete.call_args_list]
        state_kinds = [c["state_kind"] for c in delete_calls]
        assert "cooldown" in state_kinds
        assert "lane_identity" not in state_kinds  # SET keys use _delete_durable_set_key
        assert state_kinds.count("txn_receipt") == 2  # control + target
        # lane_identity SET deletion via _delete_durable_set_key.
        assert mock_set_delete.call_count == 1
        set_call = mock_set_delete.call_args_list[0].kwargs
        assert set_call["state_kind"] == "lane_identity"
        assert set_call["state_key"] == "cih"

    @pytest.mark.asyncio
    @patch(f"{_MOD}.verify_aawm_alias_routing_durable_absence", new_callable=AsyncMock)
    @patch(f"{_MOD}._delete_durable_set_key", new_callable=AsyncMock)
    @patch(f"{_MOD}.delete_aawm_alias_routing_durable_key", new_callable=AsyncMock)
    @patch(f"{_MOD}.rollback_cooldown_transaction", new_callable=AsyncMock)
    @patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
    @patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
    @patch("litellm.proxy.proxy_server", new_callable=_mock_proxy_server, create=True)
    async def test_rollback_receipt_missing_triggers_fallback(
        self, _ps, _snap, mock_dc, mock_inspect, mock_rollback, mock_delete, mock_set_delete, mock_verify, state_mgr
    ):
        """RollbackReceiptMissingError also triggers the fallback path."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import RollbackReceiptMissingError

        mock_dc.return_value = _mock_dual_cache()
        mock_inspect.return_value = _mock_inspection(exists=False, cardinality=0)
        mock_verify.return_value = True
        mock_delete.return_value = True
        mock_set_delete.return_value = None
        mock_rollback.side_effect = RollbackReceiptMissingError(
            phase="DURABLE_COMMITTED",
            family="codex",
            transaction_id_prefix="txn123",
            identity_prefix="ih",
            key_count=1,
            exception_classes=(),
        )
        self._seed_prepared()

        result = await _handle_restore(_RUN_ID, state_mgr)

        assert result["result"] == "restored"
        assert _RUN_ID not in _prepared_runs
        # 2 fallback deletions (cooldown + control receipt) + 1 target receipt.
        assert mock_delete.call_count == 3
        # 1 lane_identity SET deletion.
        assert mock_set_delete.call_count == 1

    @pytest.mark.asyncio
    @patch(f"{_MOD}._delete_durable_set_key", new_callable=AsyncMock)
    @patch(f"{_MOD}.delete_aawm_alias_routing_durable_key", new_callable=AsyncMock)
    @patch(f"{_MOD}.rollback_cooldown_transaction", new_callable=AsyncMock)
    @patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
    @patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
    @patch("litellm.proxy.proxy_server", new_callable=_mock_proxy_server, create=True)
    async def test_rollback_drift_fallback_failure_retains_state(
        self, _ps, _snap, mock_dc, mock_inspect, mock_rollback, mock_delete, mock_set_delete, state_mgr
    ):
        """When the fallback deletion itself fails, restore raises 500 and
        retains prepared state for recovery."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import RollbackDriftError

        mock_dc.return_value = _mock_dual_cache()
        mock_inspect.return_value = _mock_inspection(exists=False, cardinality=0)
        mock_rollback.side_effect = RollbackDriftError(
            phase="DURABLE_COMMITTED",
            family="codex",
            transaction_id_prefix="txn123",
            identity_prefix="ih",
            key_count=1,
            exception_classes=(),
        )
        # Fallback SET deletion fails (lane_identity is first SET deletion).
        mock_set_delete.side_effect = RuntimeError("redis unavailable")
        self._seed_prepared()

        with pytest.raises(HTTPException) as exc_info:
            await _handle_restore(_RUN_ID, state_mgr)

        detail = exc_info.value.detail
        assert detail["error"] == "control_rollback_failed"
        assert detail["phase"] == "control_rollback"
        assert detail["failure_class"] == "RuntimeError"
        assert detail["recovery_state_retained"] is True
        assert _RUN_ID in _prepared_runs

    @pytest.mark.asyncio
    @patch(f"{_MOD}.verify_aawm_alias_routing_durable_absence", new_callable=AsyncMock)
    @patch(f"{_MOD}.delete_aawm_alias_routing_durable_key", new_callable=AsyncMock)
    @patch(f"{_MOD}.rollback_cooldown_transaction", new_callable=AsyncMock)
    @patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
    @patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
    @patch("litellm.proxy.proxy_server", new_callable=_mock_proxy_server, create=True)
    async def test_identity_inspect_exception_sanitized(
        self, _ps, _snap, mock_dc, mock_inspect, mock_rollback, mock_delete, mock_verify, state_mgr
    ):
        """inspect_identity_set raising RuntimeError during postcondition
        identity inspection produces sanitized detail."""
        mock_dc.return_value = _mock_dual_cache()
        # First call (target absence proof) succeeds, second (postcondition) raises.
        mock_inspect.side_effect = [
            _mock_inspection(exists=False, cardinality=0),
            RuntimeError("Bearer secret-sentinel"),
        ]
        mock_verify.return_value = True
        self._seed_prepared()

        with pytest.raises(HTTPException) as exc_info:
            await _handle_restore(_RUN_ID, state_mgr)

        detail = exc_info.value.detail
        assert detail["error"] == "postcondition_inspect_failed"
        assert detail["phase"] == "postcondition_verification"
        assert detail["failure_class"] == "RuntimeError"
        detail_str = str(detail)
        assert "secret-sentinel" not in detail_str
        assert "Bearer" not in detail_str
        assert _RUN_ID in _prepared_runs

    @pytest.mark.asyncio
    @patch(f"{_MOD}.verify_aawm_alias_routing_durable_absence", new_callable=AsyncMock)
    @patch(f"{_MOD}.delete_aawm_alias_routing_durable_key", new_callable=AsyncMock)
    @patch(f"{_MOD}.rollback_cooldown_transaction", new_callable=AsyncMock)
    @patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
    @patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
    @patch("litellm.proxy.proxy_server", new_callable=_mock_proxy_server, create=True)
    async def test_control_receipt_verify_exception_sanitized(
        self, _ps, _snap, mock_dc, mock_inspect, mock_rollback, mock_delete, mock_verify, state_mgr
    ):
        """verify_aawm_alias_routing_durable_absence raising RuntimeError
        during control receipt verification produces sanitized detail."""
        mock_dc.return_value = _mock_dual_cache()
        mock_inspect.return_value = _mock_inspection(exists=False, cardinality=0)
        # First 2 calls (cooldown keys) succeed, 3rd (receipt) raises.
        mock_verify.side_effect = [True, True, RuntimeError("Bearer secret-sentinel")]
        self._seed_prepared()

        with pytest.raises(HTTPException) as exc_info:
            await _handle_restore(_RUN_ID, state_mgr)

        detail = exc_info.value.detail
        assert detail["error"] == "control_receipt_verify_failed"
        assert detail["phase"] == "postcondition_verification"
        assert detail["failure_class"] == "RuntimeError"
        detail_str = str(detail)
        assert "secret-sentinel" not in detail_str
        assert "Bearer" not in detail_str
        assert _RUN_ID in _prepared_runs

    @pytest.mark.asyncio
    @patch(f"{_MOD}.verify_aawm_alias_routing_durable_absence", new_callable=AsyncMock)
    @patch(f"{_MOD}.delete_aawm_alias_routing_durable_key", new_callable=AsyncMock)
    @patch(f"{_MOD}.rollback_cooldown_transaction", new_callable=AsyncMock)
    @patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
    @patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
    @patch("litellm.proxy.proxy_server", new_callable=_mock_proxy_server, create=True)
    async def test_target_receipt_deletion_exception_sanitized(
        self, _ps, _snap, mock_dc, mock_inspect, mock_rollback, mock_delete, mock_verify, state_mgr
    ):
        """delete_aawm_alias_routing_durable_key raising RuntimeError
        during target receipt cleanup produces sanitized detail."""
        mock_dc.return_value = _mock_dual_cache()
        mock_inspect.return_value = _mock_inspection(exists=False, cardinality=0)
        mock_verify.return_value = True
        mock_delete.side_effect = RuntimeError("Bearer secret-sentinel")
        self._seed_prepared()

        with pytest.raises(HTTPException) as exc_info:
            await _handle_restore(_RUN_ID, state_mgr)

        detail = exc_info.value.detail
        assert detail["error"] == "target_receipt_deletion_failed"
        assert detail["phase"] == "target_receipt_cleanup"
        assert detail["failure_class"] == "RuntimeError"
        assert detail["target_receipt_retained"] is True
        detail_str = str(detail)
        assert "secret-sentinel" not in detail_str
        assert "Bearer" not in detail_str
        assert _RUN_ID in _prepared_runs

    @pytest.mark.asyncio
    @patch(f"{_MOD}.verify_aawm_alias_routing_durable_absence", new_callable=AsyncMock)
    @patch(f"{_MOD}.delete_aawm_alias_routing_durable_key", new_callable=AsyncMock)
    @patch(f"{_MOD}.rollback_cooldown_transaction", new_callable=AsyncMock)
    @patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
    @patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
    @patch("litellm.proxy.proxy_server", new_callable=_mock_proxy_server, create=True)
    async def test_target_receipt_verify_exception_sanitized(
        self, _ps, _snap, mock_dc, mock_inspect, mock_rollback, mock_delete, mock_verify, state_mgr
    ):
        """verify_aawm_alias_routing_durable_absence raising RuntimeError
        during target receipt post-deletion verification produces sanitized
        detail with target_receipt_retained=False."""
        mock_dc.return_value = _mock_dual_cache()
        mock_inspect.return_value = _mock_inspection(exists=False, cardinality=0)
        mock_delete.return_value = True
        # All postcondition verifies pass, then target receipt verify raises.
        # 2 cooldown keys + 1 control receipt = 3 successful, then raise.
        mock_verify.side_effect = [True, True, True, RuntimeError("Bearer secret-sentinel")]
        self._seed_prepared()

        with pytest.raises(HTTPException) as exc_info:
            await _handle_restore(_RUN_ID, state_mgr)

        detail = exc_info.value.detail
        assert detail["error"] == "target_receipt_verify_failed"
        assert detail["phase"] == "target_receipt_cleanup"
        assert detail["failure_class"] == "RuntimeError"
        assert detail["target_receipt_retained"] is False
        detail_str = str(detail)
        assert "secret-sentinel" not in detail_str
        assert "Bearer" not in detail_str
        assert _RUN_ID in _prepared_runs

    @pytest.mark.asyncio
    @patch(f"{_MOD}.verify_aawm_alias_routing_durable_absence", new_callable=AsyncMock)
    @patch(f"{_MOD}.delete_aawm_alias_routing_durable_key", new_callable=AsyncMock)
    @patch(f"{_MOD}.rollback_cooldown_transaction", new_callable=AsyncMock)
    @patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
    @patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
    @patch("litellm.proxy.proxy_server", new_callable=_mock_proxy_server, create=True)
    async def test_control_rollback_exception_sanitized(
        self, _ps, _snap, mock_dc, mock_inspect, mock_rollback, mock_delete, mock_verify, state_mgr
    ):
        """rollback_cooldown_transaction raising RuntimeError('Bearer secret-sentinel')
        produces sanitized detail without the secret."""
        mock_dc.return_value = _mock_dual_cache()
        mock_inspect.return_value = _mock_inspection(exists=False, cardinality=0)
        mock_rollback.side_effect = RuntimeError("Bearer secret-sentinel")
        self._seed_prepared()

        with pytest.raises(HTTPException) as exc_info:
            await _handle_restore(_RUN_ID, state_mgr)

        detail = exc_info.value.detail
        assert detail["error"] == "control_rollback_failed"
        assert detail["phase"] == "control_rollback"
        assert detail["failure_class"] == "RuntimeError"
        detail_str = str(detail)
        assert "secret-sentinel" not in detail_str
        assert "Bearer" not in detail_str
        assert _RUN_ID in _prepared_runs


# ---------------------------------------------------------------------------
# Codex CLI 0.146+ response_item collaboration schema normalization
# ---------------------------------------------------------------------------


def _codex_0146_collaboration_stdout(
    *,
    include_spawn: bool = True,
    include_wait: bool = True,
    include_child_exec: bool = False,
    wait_timed_out: bool = False,
) -> str:
    """Build sanitized JSONL stdout matching the Codex CLI 0.146 parent
    transcript shape: ``response_item`` payloads with ``type=function_call``,
    ``namespace=collaboration``, plus correlated ``function_call_output``."""
    import json as _json

    lines: list[str] = []
    lines.append(_json.dumps({"type": "turn.started", "turn_id": "t1"}))

    if include_spawn:
        lines.append(_json.dumps({
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "id": "fc_spawn_01",
                "name": "spawn_agent",
                "namespace": "collaboration",
                "arguments": '{"task_name":"basic_alias_child"}',
                "call_id": "call_spawn_01",
            },
        }))
        lines.append(_json.dumps({
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "id": "fco_spawn_01",
                "call_id": "call_spawn_01",
                "output": '{"task_name":"/root/basic_alias_child"}',
            },
        }))

    if include_child_exec:
        lines.append(_json.dumps({
            "type": "item.started",
            "item": {
                "type": "command_execution",
                "id": "cmd_01",
                "command": "echo hello",
                "status": "in_progress",
            },
        }))
        lines.append(_json.dumps({
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "id": "cmd_01",
                "command": "echo hello",
                "status": "completed",
                "exit_code": 0,
            },
        }))

    if include_wait:
        lines.append(_json.dumps({
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "id": "fc_wait_01",
                "name": "wait_agent",
                "namespace": "collaboration",
                "arguments": '{"timeout_ms":3600000}',
                "call_id": "call_wait_01",
            },
        }))
        wait_output = _json.dumps({
            "message": "Wait completed.",
            "timed_out": wait_timed_out,
            "agents_states": {
                "/root/basic_alias_child": {"status": "completed"},
            },
        })
        lines.append(_json.dumps({
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "id": "fco_wait_01",
                "call_id": "call_wait_01",
                "output": wait_output,
            },
        }))

    return "\n".join(lines)


class TestCodexCollaborationSchemaNormalization:
    """Focused tests for the Codex CLI 0.146+ response_item collaboration
    event normalization in ``_validate_codex_collaboration_events``."""

    @pytest.fixture(autouse=True)
    def _load_script(self):
        self.script = _load_acceptance_script()

    def test_spawn_and_wait_counted_from_response_item_schema(self):
        """spawn_agent and wait_agent from the new schema are counted and
        pass minimum_tool_counts and required_successful_tools."""
        stdout = _codex_0146_collaboration_stdout()
        checks = {
            "minimum_tool_counts": {"spawn_agent": 1, "wait": 1},
            "required_successful_tools": ["spawn_agent", "wait"],
            "require_wait_for_spawned_agents": True,
        }
        summary, failures = self.script._validate_codex_collaboration_events(
            family="test", stdout=stdout, checks=checks,
        )
        assert failures == [], f"Unexpected failures: {failures}"
        assert summary["tool_counts"]["spawn_agent"] >= 1
        assert summary["tool_counts"]["wait"] >= 1
        assert "/root/basic_alias_child" in summary["spawned_agent_ids"]
        assert summary["waited_agent_statuses"].get("/root/basic_alias_child") == "completed"

    def test_absent_child_exec_command_remains_failure(self):
        """When command_execution_validation expects a child command that is
        absent from the stdout, validation still fails even though collab
        events parse correctly."""
        stdout = _codex_0146_collaboration_stdout(include_child_exec=False)
        checks = {
            "minimum_tool_counts": {"spawn_agent": 1, "wait": 1},
            "required_successful_tools": ["spawn_agent", "wait"],
            "command_execution_validation": {
                "exact_commands": ["echo hello"],
            },
        }
        summary, failures = self.script._validate_codex_collaboration_events(
            family="test", stdout=stdout, checks=checks,
        )
        assert any("unexpected commands" in f for f in failures)

    def test_child_exec_command_present_passes(self):
        """When the child command_execution events are present, the full
        validation passes with both schemas active."""
        stdout = _codex_0146_collaboration_stdout(include_child_exec=True)
        checks = {
            "minimum_tool_counts": {"spawn_agent": 1, "wait": 1},
            "required_successful_tools": ["spawn_agent", "wait"],
            "command_execution_validation": {
                "exact_commands": ["echo hello"],
                "required_status": "completed",
                "required_exit_code": 0,
            },
        }
        summary, failures = self.script._validate_codex_collaboration_events(
            family="test", stdout=stdout, checks=checks,
        )
        assert failures == [], f"Unexpected failures: {failures}"
        assert summary["command_execution"]["enabled"] is True

    def test_wait_timed_out_fails(self):
        """A wait_agent output with timed_out=true produces a failure."""
        stdout = _codex_0146_collaboration_stdout(wait_timed_out=True)
        checks = {
            "minimum_tool_counts": {"spawn_agent": 1, "wait": 1},
            "required_successful_tools": ["wait"],
        }
        summary, failures = self.script._validate_codex_collaboration_events(
            family="test", stdout=stdout, checks=checks,
        )
        assert any("did not complete successfully" in f for f in failures)

    def test_missing_wait_for_spawned_agents_fails(self):
        """require_wait_for_spawned_agents catches spawn without wait."""
        stdout = _codex_0146_collaboration_stdout(include_wait=False)
        checks = {
            "minimum_tool_counts": {"spawn_agent": 1},
            "required_successful_tools": ["spawn_agent"],
            "require_wait_for_spawned_agents": True,
        }
        summary, failures = self.script._validate_codex_collaboration_events(
            family="test", stdout=stdout, checks=checks,
        )
        assert any("did not report spawned agents" in f for f in failures)

    def test_old_schema_still_recognized(self):
        """The older item.completed / collab_tool_call schema continues to
        work alongside the new response_item schema."""
        import json as _json

        lines = [
            _json.dumps({"type": "turn.started", "turn_id": "t1"}),
            _json.dumps({
                "type": "item.completed",
                "item": {
                    "type": "collab_tool_call",
                    "tool": "functions.collaboration.spawn_agent",
                    "status": "completed",
                    "receiver_agent_ids": ["agent-x"],
                    "result": {},
                },
            }),
            _json.dumps({
                "type": "item.completed",
                "item": {
                    "type": "collab_tool_call",
                    "tool": "functions.collaboration.wait",
                    "status": "completed",
                    "agents_states": {"agent-x": {"status": "completed"}},
                    "result": {},
                },
            }),
        ]
        stdout = "\n".join(lines)
        checks = {
            "minimum_tool_counts": {"spawn_agent": 1, "wait": 1},
            "required_successful_tools": ["spawn_agent", "wait"],
            "require_wait_for_spawned_agents": True,
        }
        summary, failures = self.script._validate_codex_collaboration_events(
            family="test", stdout=stdout, checks=checks,
        )
        assert failures == [], f"Unexpected failures: {failures}"
        assert summary["tool_counts"]["spawn_agent"] == 1
        assert summary["tool_counts"]["wait"] == 1

    def test_no_collab_events_zero_counts(self):
        """Empty stdout produces zero tool counts and fails minimums."""
        checks = {
            "minimum_tool_counts": {"spawn_agent": 1},
        }
        summary, failures = self.script._validate_codex_collaboration_events(
            family="test", stdout="", checks=checks,
        )
        assert any("missing completed Codex collaboration calls" in f for f in failures)
        assert summary["tool_counts"].get("spawn_agent", 0) == 0


class TestCfg003PhaseSessionCorrelation:
    """Prove _cfg003_run_proof_case rerenders the Codex command's
    http_headers.session_id to match the fresh phase_session_id, so the
    rendered command header, expected trace/session ID, session_history
    query, and tool_activity query all correlate on the same ID."""

    @pytest.fixture(autouse=True)
    def _load_script(self):
        self.script = _load_acceptance_script()

    def test_phase_session_id_correlates_command_and_queries(self):
        old_session = "old-profile-session-id"
        captured_config: dict = {}

        def _capture_run_selected_case(**kwargs):
            captured_config.update(kwargs.get("case_config", {}))
            return {"passed": True, "langfuse": {"command_session_id": "x"}}

        case_config = {
            "cli_passthrough": "codex",
            "command": [
                "codex", "exec",
                "-c", f'model_providers.litellm-dev.http_headers.session_id="{old_session}"',
                "-c", 'model_providers.litellm-dev.http_headers.x-aawm-tenant-id="tenant"',
                "--json",
                "do the thing",
            ],
            "expected_trace_session_id": old_session,
            "match_trace_session_id_from_stdout": False,
            "session_history_validation": {"minimum_rows": 1},
            "tool_activity_validation": {"minimum_rows": 1},
        }
        cases = {"proof_case": case_config}

        with patch.object(
            self.script, "_run_selected_case", side_effect=_capture_run_selected_case
        ):
            proof = self.script._cfg003_run_proof_case(
                case_name="proof_case",
                case_config_key="proof_case",
                cases=cases,
                suite_config={},
                query_url="http://localhost:3000",
                public_key="pk",
                secret_key="sk",
                litellm_base_url="http://127.0.0.1:4001",
            )

        phase_id = proof["phase_session_id"]
        assert phase_id != old_session, "phase ID must be fresh"

        # 1. Rendered command header carries the fresh phase ID.
        cmd = captured_config["command"]
        session_args = [
            a for a in cmd
            if str(a).startswith("model_providers.") and ".http_headers.session_id=" in str(a)
        ]
        assert len(session_args) == 1
        assert f'"{phase_id}"' in str(session_args[0])
        assert old_session not in str(session_args[0])

        # Other header args are untouched.
        tenant_args = [
            a for a in cmd
            if "x-aawm-tenant-id" in str(a)
        ]
        assert len(tenant_args) == 1
        assert '"tenant"' in str(tenant_args[0])

        # 2. expected_trace_session_id matches the fresh phase ID.
        assert captured_config["expected_trace_session_id"] == phase_id

        # 3. session_history_validation has phase_start_time injected.
        shv = captured_config["session_history_validation"]
        assert "phase_start_time" in shv

        # 4. tool_activity_validation has phase_start_time injected.
        tav = captured_config["tool_activity_validation"]
        assert "phase_start_time" in tav

        # 5. With match_trace_session_id_from_stdout=False, the validation
        #    path uses expected_trace_session_id (== phase_id) for DB queries.
        assert captured_config["match_trace_session_id_from_stdout"] is False
