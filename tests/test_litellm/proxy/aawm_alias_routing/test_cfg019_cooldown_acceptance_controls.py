"""CFG-019 acceptance controls for generic alias-routing cooldown validation.

Covers only seed_durable, inspect_durable, and request_local_seed on the
existing gated acceptance surface. No provider calls.

CFG-019 follow-up findings covered here:
1. Configured OpenAI OAuth lane parity is proven through the real
   resolver/handler path (no normalization bypass).
2. ``inspect_durable`` can inspect/prove invalidation for a previously
   seeded identity after candidate semantic change or removal, via a
   restart-safe sanitized prior-identity handle.
3. ``seed_durable`` retains the real ``publish_cooldown_transaction``
   receipt/journal and rolls back ONLY that transaction when bounded
   post-seed TTL verification fails.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    cooldown_acceptance as ca_mod,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
    DEFAULT_CONFIG_DIR,
    compile_directory,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_snapshot import (
    RoutingAlias,
    RoutingCandidate,
    RoutingSnapshot,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_acceptance import (
    UNBOUNDED_EXPIRY,
    _handle_inspect_durable,
    _handle_request_local_seed,
    _handle_seed_durable,
    _prepared_runs,
    _resolve_cfg019_openai_lane_key,
    _resolve_eligible_candidates,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    AliasRoutingStateManager,
)

_MOD = "litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_acceptance"
_RUN_ID = "a" * 32
_TARGET_PROVIDER = "openai"
_TARGET_MODEL = "gpt-5.3-codex"


_CODEX_OAUTH_ACCOUNT_ID = "01234567-89ab-cdef-0123-456789abcdef"
_CONFIGURED_OAUTH_LANE = f"chatgpt-account:{_CODEX_OAUTH_ACCOUNT_ID}"


def _make_snapshot() -> RoutingSnapshot:
    target = RoutingCandidate(
        provider=_TARGET_PROVIDER,
        model=_TARGET_MODEL,
        route_family="codex_openai_responses",
        priority=10,
        weight=1.0,
        tui_attached=None,
        schedule=None,
    )
    alias = RoutingAlias(
        name="basic",
        distribution_strategy=None,
        candidates=(target,),
    )
    return RoutingSnapshot(
        aliases={"basic": alias},
        config_epoch=1,
        config_hash="abcdef123456",
        config_version="abcdef123456",
    )


def _make_request(*, auth_header: str = "Bearer sk-test-auth") -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/aawm/alias-routing/cooldowns/acceptance",
        "headers": [(b"authorization", auth_header.encode("latin-1"))],
        "query_string": b"",
    }
    return Request(scope)


def _seed_body(*, ttl_seconds: float = 300) -> dict[str, object]:
    return {
        "operation": "seed_durable",
        "run_id": _RUN_ID,
        "alias": "basic",
        "ingress": "codex",
        "provider": _TARGET_PROVIDER,
        "model": _TARGET_MODEL,
        "ttl_seconds": ttl_seconds,
    }


def _inspect_body() -> dict[str, object]:
    return {
        "operation": "inspect_durable",
        "run_id": _RUN_ID,
        "alias": "basic",
        "ingress": "codex",
        "provider": _TARGET_PROVIDER,
        "model": _TARGET_MODEL,
    }


def _request_local_body(*, ttl_seconds: float = 300) -> dict[str, object]:
    return {
        "operation": "request_local_seed",
        "run_id": _RUN_ID,
        "alias": "basic",
        "ingress": "codex",
        "provider": _TARGET_PROVIDER,
        "model": _TARGET_MODEL,
        "ttl_seconds": ttl_seconds,
    }


def _mock_dual_cache():
    dc = MagicMock()
    dc.redis_cache = MagicMock()
    dc.redis_cache.init_async_client = MagicMock(return_value=MagicMock())
    return dc


def _mock_identity_inspection(*, exists: bool = True, cardinality: int = 1, members=None):
    insp = MagicMock()
    insp.exists = exists
    insp.cardinality = cardinality
    insp.members = frozenset() if members is None else members
    insp.ttl_remaining_seconds = None
    return insp


def _mock_key_inspection(*, exists: bool = True, ttl_seconds: float = 120.0):
    insp = MagicMock()
    insp.exists = exists
    insp.ttl_remaining_seconds = ttl_seconds
    return insp


def _fake_safe_get_request_headers(request):
    if request is None:
        return {}
    headers = getattr(request, "headers", None)
    if isinstance(headers, dict):
        return headers
    try:
        return {key: value for key, value in headers.items()}
    except Exception:
        return {}


def _fake_clean_codex_auth_value(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower().startswith("bearer "):
        text = text[7:].strip()
    return text or None


def _make_txn_result(*, cooldown_key: str, identity_hash: str, transaction_id: str = "txn-abc"):
    """Build a real CooldownTransactionResult + journal for one seeded key."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        CooldownTransactionJournal,
        CooldownTransactionResult,
    )

    journal = CooldownTransactionJournal(
        transaction_id=transaction_id,
        phase="DURABLE_COMMITTED",
        alias_family="codex",
        identity_hash=identity_hash,
        cooldown_keys=[cooldown_key],
        identity_keys=[f"lane-identity:{identity_hash}"],
        lane_members=[cooldown_key],
        preimages=[(None, -2)],
        receipt_key=f"txn-receipt:{transaction_id}",
        requested_ttl=120,
    )
    return CooldownTransactionResult(
        transaction_id=transaction_id,
        phase="DURABLE_COMMITTED",
        journal=journal,
    )


@pytest.fixture(autouse=True)
def _clear_prepared():
    _prepared_runs.clear()
    yield
    _prepared_runs.clear()


@pytest.fixture(autouse=True)
def _patch_lane_keys_headers():
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


def _assert_sanitized_public_fields(result: dict[str, object]) -> None:
    for field in ("identity_hash", "lane_hash", "config_hash"):
        value = result[field]
        assert isinstance(value, str)
        assert len(value) == 64
        assert all(char in "0123456789abcdef" for char in value)
    assert "provider" not in result
    assert "model" not in result
    assert "route_family" not in result
    result_str = str(result)
    assert _TARGET_PROVIDER not in result_str
    assert _TARGET_MODEL not in result_str
    assert "codex_openai_responses" not in result_str
    assert "sk-test-auth" not in result_str
    assert "cooldown_until_monotonic_by_key" not in result_str
    assert _CODEX_OAUTH_ACCOUNT_ID not in result_str
    assert _CONFIGURED_OAUTH_LANE not in result_str


@pytest.mark.asyncio
@patch(f"{_MOD}.inspect_aawm_alias_routing_durable_key", new_callable=AsyncMock)
@patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
@patch(f"{_MOD}.publish_cooldown_transaction", new_callable=AsyncMock)
@patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
@patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
async def test_seed_durable_writes_redis_only(
    _snap,
    mock_dc,
    mock_publish,
    mock_inspect,
    mock_key_inspect,
):
    mock_dc.return_value = _mock_dual_cache()
    mock_publish.return_value = MagicMock()
    mock_inspect.side_effect = [
        _mock_identity_inspection(exists=False, cardinality=0),
        _mock_identity_inspection(exists=True, cardinality=1),
    ]
    mock_key_inspect.side_effect = [
        _mock_key_inspection(exists=False, ttl_seconds=0.0),
        _mock_key_inspection(ttl_seconds=119.0),
    ]
    state_mgr = AliasRoutingStateManager()
    request = _make_request()

    result = await _handle_seed_durable(_seed_body(), _RUN_ID, request, state_mgr)

    assert result["result"] == "seeded_durable"
    assert result["attempted_provider_call"] is False
    assert result["durable_cooldown_active"] is True
    assert result["local_cooldown_active"] is False
    assert result["request_local_cooldown_active"] is False
    assert result["prepared_run_stored"] is False
    assert result["state_source"] == "durable_cache"
    assert result["ttl_seconds"] == 119.0
    _assert_sanitized_public_fields(result)
    mock_publish.assert_awaited_once()
    assert _RUN_ID not in _prepared_runs
    assert state_mgr.codex.cooldown_until_monotonic_by_key == {}
    assert len(state_mgr.lane_identity_index) == 0
    assert not hasattr(request.state, "aawm_alias_request_local_cooldown_until") or (
        getattr(request.state, "aawm_alias_request_local_cooldown_until", {}) == {}
    )


@pytest.mark.asyncio
@patch(f"{_MOD}.inspect_aawm_alias_routing_durable_key", new_callable=AsyncMock)
@patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
@patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
@patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
async def test_inspect_durable_is_restart_safe_without_prepared_runs(
    _snap,
    mock_dc,
    mock_inspect,
    mock_key_inspect,
):
    mock_dc.return_value = _mock_dual_cache()
    mock_inspect.return_value = _mock_identity_inspection()
    mock_key_inspect.return_value = _mock_key_inspection(ttl_seconds=88.0)
    request = _make_request()
    assert _prepared_runs == {}

    result = await _handle_inspect_durable(_inspect_body(), _RUN_ID, request)

    assert result["result"] == "inspected_durable"
    assert result["attempted_provider_call"] is False
    assert result["durable_cooldown_active"] is True
    assert result["prepared_run_required"] is False
    assert result["state_source"] == "durable_cache"
    assert result["ttl_seconds"] == 88.0
    _assert_sanitized_public_fields(result)
    assert _prepared_runs == {}
    mock_inspect.assert_awaited_once()
    mock_key_inspect.assert_awaited_once()


@pytest.mark.asyncio
@patch(f"{_MOD}.inspect_aawm_alias_routing_durable_key", new_callable=AsyncMock)
@patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
@patch(f"{_MOD}.publish_cooldown_transaction", new_callable=AsyncMock)
@patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
@patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
async def test_request_local_seed_mutates_request_state_only(
    _snap,
    mock_dc,
    mock_publish,
    mock_inspect,
    mock_key_inspect,
):
    mock_dc.return_value = _mock_dual_cache()
    state_mgr = AliasRoutingStateManager()
    request = _make_request()

    result = await _handle_request_local_seed(
        _request_local_body(ttl_seconds=45),
        _RUN_ID,
        request,
        state_mgr,
    )

    assert result["result"] == "seeded_request_local"
    assert result["attempted_provider_call"] is False
    assert result["durable_cooldown_active"] is False
    assert result["local_cooldown_active"] is False
    assert result["request_local_cooldown_active"] is True
    assert result["prepared_run_stored"] is False
    assert result["state_source"] == "request_local"
    assert result["ttl_seconds"] > 44.0
    _assert_sanitized_public_fields(result)
    request_local_state = getattr(
        request.state, "aawm_alias_request_local_cooldown_until", {}
    )
    assert request_local_state
    assert all(until > 0 for until in request_local_state.values())
    mock_publish.assert_not_called()
    mock_inspect.assert_not_called()
    mock_key_inspect.assert_not_called()
    mock_dc.assert_not_called()
    assert _RUN_ID not in _prepared_runs
    assert state_mgr.codex.cooldown_until_monotonic_by_key == {}
    assert len(state_mgr.lane_identity_index) == 0


@pytest.mark.asyncio
@patch(f"{_MOD}.inspect_aawm_alias_routing_durable_key", new_callable=AsyncMock)
@patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
@patch(f"{_MOD}.publish_cooldown_transaction", new_callable=AsyncMock)
@patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
@patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
async def test_new_operations_reject_unsupported_provider(
    _snap,
    mock_dc,
    mock_publish,
    mock_inspect,
    mock_key_inspect,
):
    mock_dc.return_value = _mock_dual_cache()
    state_mgr = AliasRoutingStateManager()
    request = _make_request()
    body = _seed_body()
    body["provider"] = "cohere"
    body["model"] = "command-a"

    with pytest.raises(HTTPException) as exc_info:
        await _handle_seed_durable(body, _RUN_ID, request, state_mgr)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["error"] == "unsupported_provider"
    mock_publish.assert_not_called()
    mock_inspect.assert_not_called()
    mock_key_inspect.assert_not_called()


def test_cfg019_oauth_lane_matches_configured_or_validated_descriptor():
    request = _make_request()
    configured = {
        "provider": "openai",
        "model": _TARGET_MODEL,
        "codex_oauth_lane_key": _CONFIGURED_OAUTH_LANE,
    }
    assert (
        _resolve_cfg019_openai_lane_key(
            request,
            configured,
            codex_oauth_account_id=None,
        )
        == _CONFIGURED_OAUTH_LANE
    )
    validated = {
        "provider": "openai",
        "model": _TARGET_MODEL,
    }
    assert (
        _resolve_cfg019_openai_lane_key(
            request,
            validated,
            codex_oauth_account_id=_CODEX_OAUTH_ACCOUNT_ID,
        )
        == _CONFIGURED_OAUTH_LANE
    )


@pytest.mark.asyncio
@patch(f"{_MOD}.inspect_aawm_alias_routing_durable_key", new_callable=AsyncMock)
@patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
@patch(f"{_MOD}.publish_cooldown_transaction", new_callable=AsyncMock)
@patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
@patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
async def test_new_operation_responses_omit_raw_provider_model_route(
    _snap,
    mock_dc,
    mock_publish,
    mock_inspect,
    mock_key_inspect,
):
    mock_dc.return_value = _mock_dual_cache()
    mock_inspect.return_value = _mock_identity_inspection()
    mock_key_inspect.return_value = _mock_key_inspection(ttl_seconds=77.0)
    request = _make_request()

    result = await _handle_inspect_durable(_inspect_body(), _RUN_ID, request)

    _assert_sanitized_public_fields(result)
    assert result["result"] == "inspected_durable"
    mock_publish.assert_not_called()


@pytest.mark.asyncio
@patch(f"{_MOD}.inspect_aawm_alias_routing_durable_key", new_callable=AsyncMock)
@patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
@patch(f"{_MOD}.publish_cooldown_transaction", new_callable=AsyncMock)
@patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
@patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
async def test_seed_durable_rejects_existing_unbounded_prestate(
    _snap,
    mock_dc,
    mock_publish,
    mock_inspect,
    mock_key_inspect,
):
    mock_dc.return_value = _mock_dual_cache()
    mock_inspect.return_value = _mock_identity_inspection(exists=True, cardinality=1)
    unbounded = _mock_key_inspection(exists=True, ttl_seconds=0.0)
    unbounded.ttl_remaining_seconds = UNBOUNDED_EXPIRY
    unbounded.payload = {"persistent": True}
    mock_key_inspect.return_value = unbounded
    state_mgr = AliasRoutingStateManager()
    request = _make_request()

    with pytest.raises(HTTPException) as exc_info:
        await _handle_seed_durable(_seed_body(), _RUN_ID, request, state_mgr)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["error"] == "prestate_unbounded"
    mock_publish.assert_not_called()


@pytest.mark.asyncio
@patch(f"{_MOD}.rollback_cooldown_transaction", new_callable=AsyncMock)
@patch(f"{_MOD}.inspect_aawm_alias_routing_durable_key", new_callable=AsyncMock)
@patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
@patch(f"{_MOD}.publish_cooldown_transaction", new_callable=AsyncMock)
@patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
@patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
async def test_seed_durable_requires_finite_bounded_postcondition(
    _snap,
    mock_dc,
    mock_publish,
    mock_inspect,
    mock_key_inspect,
    mock_rollback,
):
    mock_dc.return_value = _mock_dual_cache()
    async def _fake_publish(*, alias_family, identity_hash, cooldown_keys, lane_members, ttl_seconds, **kw):
        return _make_txn_result(
            cooldown_key=cooldown_keys[0], identity_hash=identity_hash
        )

    mock_publish.side_effect = _fake_publish
    mock_inspect.side_effect = [
        _mock_identity_inspection(exists=False, cardinality=0),
        _mock_identity_inspection(exists=True, cardinality=1),
    ]
    post = _mock_key_inspection(exists=True, ttl_seconds=0.0)
    post.ttl_remaining_seconds = UNBOUNDED_EXPIRY
    mock_key_inspect.side_effect = [
        _mock_key_inspection(exists=False, ttl_seconds=0.0),
        post,
    ]
    state_mgr = AliasRoutingStateManager()
    request = _make_request()

    with pytest.raises(HTTPException) as exc_info:
        await _handle_seed_durable(_seed_body(ttl_seconds=120), _RUN_ID, request, state_mgr)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail["error"] == "seed_verification_failed"
    mock_publish.assert_awaited_once()
    # The seeded transaction is rolled back on post-seed TTL failure.
    mock_rollback.assert_awaited_once()


# ---------------------------------------------------------------------------
# Finding 1: configured OpenAI OAuth lane parity through the real
# resolver/handler path (no normalization bypass).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _canonical_openai_candidate():
    """Compile the real canonical config and resolve the ``basic`` OpenAI
    candidate through the real resolver (``_resolve_snapshot_alias_candidates``)
    without bypassing candidate normalization."""
    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    canonical_alias, candidates = None, None
    with patch(f"{_MOD}.get_active_routing_snapshot", return_value=snapshot):
        canonical_alias, candidates = _resolve_eligible_candidates("basic")
    assert canonical_alias == "basic"
    openai = [c for c in candidates if c["provider"] == "openai"]
    assert openai, "expected an OpenAI candidate in the canonical basic alias"
    return snapshot, openai[0]


def test_candidate_normalization_preserves_oauth_lane_key(_canonical_openai_candidate):
    """`_resolve_eligible_candidates` must carry a configured managed OAuth
    lane key through normalization rather than dropping it."""
    snapshot, _ = _canonical_openai_candidate
    lane = _CONFIGURED_OAUTH_LANE

    shaped = {
        "provider": "openai",
        "model": _TARGET_MODEL,
        "route_family": "codex_responses",
        "config_epoch_tag": snapshot.config_hash,
        "cooldown_identity_tag": (
            f"alias:basic:openai:{_TARGET_MODEL}:codex_responses"
        ),
        "codex_oauth_lane_key": lane,
    }
    with patch(
        f"{_MOD}.get_active_routing_snapshot", return_value=snapshot
    ), patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select._resolve_snapshot_alias_candidates",
        return_value=[shaped],
    ):
        _, candidates = _resolve_eligible_candidates("basic")

    assert len(candidates) == 1
    assert candidates[0]["codex_oauth_lane_key"] == lane


@pytest.mark.asyncio
async def test_configured_oauth_lane_parity_through_real_handler(
    _canonical_openai_candidate,
):
    """Prove parity: seed_durable publishes the configured managed OAuth lane
    that the real production selector would use, through the real
    resolver/handler path with no normalization bypass.

    Parity is proven by independently deriving the expected lane via the real
    production helper ``_resolve_cfg019_openai_lane_key`` on the same
    resolver-normalized candidate and asserting the published lane key equals
    it (and is NOT the auth-derived fallback)."""
    snapshot, resolved_candidate = _canonical_openai_candidate

    # Attach the configured managed OAuth lane the way production
    # `_resolve_codex_oauth_account_candidate_contexts` does.
    managed = dict(resolved_candidate)
    managed["codex_oauth_lane_key"] = _CONFIGURED_OAUTH_LANE

    # Expected lane via the real production lane-selection helper on the same
    # resolver-normalized candidate.
    request = _make_request()
    expected_lane = _resolve_cfg019_openai_lane_key(
        request, managed, codex_oauth_account_id=None,
    )
    assert expected_lane == _CONFIGURED_OAUTH_LANE

    with patch(f"{_MOD}.get_active_routing_snapshot", return_value=snapshot), patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select._resolve_snapshot_alias_candidates",
        return_value=[managed],
    ), patch(f"{_MOD}.get_aawm_alias_routing_dual_cache") as mock_dc, patch(
        f"{_MOD}.publish_cooldown_transaction", new_callable=AsyncMock
    ) as mock_publish, patch(
        f"{_MOD}.inspect_identity_set", new_callable=AsyncMock
    ) as mock_inspect, patch(
        f"{_MOD}.inspect_aawm_alias_routing_durable_key", new_callable=AsyncMock
    ) as mock_key_inspect:
        mock_dc.return_value = _mock_dual_cache()

        async def _fake_publish(*, alias_family, identity_hash, cooldown_keys, lane_members, ttl_seconds, **kw):
            return _make_txn_result(
                cooldown_key=cooldown_keys[0], identity_hash=identity_hash
            )

        mock_publish.side_effect = _fake_publish
        mock_inspect.side_effect = [
            _mock_identity_inspection(exists=False, cardinality=0),
            _mock_identity_inspection(exists=True, cardinality=1),
        ]
        mock_key_inspect.side_effect = [
            _mock_key_inspection(exists=False, ttl_seconds=0.0),
            _mock_key_inspection(ttl_seconds=119.0),
        ]
        state_mgr = AliasRoutingStateManager()
        body = _seed_body(ttl_seconds=300)
        body["model"] = managed["model"]

        result = await _handle_seed_durable(body, _RUN_ID, request, state_mgr)

    assert result["result"] == "seeded_durable"
    assert result["attempted_provider_call"] is False
    mock_publish.assert_awaited_once()
    published_lane = mock_publish.await_args.kwargs["cooldown_keys"][0]
    # Parity: the published lane key embeds the configured managed OAuth lane,
    # not the auth-derived request fallback.
    assert expected_lane in published_lane
    assert "auth:" not in published_lane
    _assert_sanitized_public_fields(result)


# ---------------------------------------------------------------------------
# Finding 2: restart-safe prior-identity inspection after semantic change /
# removal.
# ---------------------------------------------------------------------------


def _prior_snapshot() -> RoutingSnapshot:
    """Snapshot whose OpenAI candidate has the ORIGINAL route family."""
    target = RoutingCandidate(
        provider=_TARGET_PROVIDER,
        model=_TARGET_MODEL,
        route_family="codex_openai_responses",
        priority=10,
        weight=1.0,
        tui_attached=None,
        schedule=None,
    )
    alias = RoutingAlias(
        name="basic",
        distribution_strategy=None,
        candidates=(target,),
    )
    return RoutingSnapshot(
        aliases={"basic": alias},
        config_epoch=1,
        config_hash="aaaa1111",
        config_version="aaaa1111",
    )


def _changed_snapshot() -> RoutingSnapshot:
    """Snapshot where the same provider/model changed route semantics."""
    target = RoutingCandidate(
        provider=_TARGET_PROVIDER,
        model=_TARGET_MODEL,
        route_family="codex_openai_chat_completions_adapter",
        priority=10,
        weight=1.0,
        tui_attached=None,
        schedule=None,
    )
    alias = RoutingAlias(
        name="basic",
        distribution_strategy=None,
        candidates=(target,),
    )
    return RoutingSnapshot(
        aliases={"basic": alias},
        config_epoch=2,
        config_hash="bbbb2222",
        config_version="bbbb2222",
    )


def _removed_snapshot() -> RoutingSnapshot:
    """Snapshot whose previously seeded OpenAI target is genuinely gone."""
    leftover = RoutingCandidate(
        provider=_TARGET_PROVIDER,
        model="gpt-4.1",
        route_family="codex_openai_responses",
        priority=10,
        weight=1.0,
        tui_attached=None,
        schedule=None,
    )
    alias = RoutingAlias(
        name="basic",
        distribution_strategy=None,
        candidates=(leftover,),
    )
    return RoutingSnapshot(
        aliases={"basic": alias},
        config_epoch=3,
        config_hash="cccc3333",
        config_version="cccc3333",
    )


def _inspect_prior_body(*, route_family: str) -> dict[str, object]:
    body = _inspect_body()
    body["route_family"] = route_family
    return body


@pytest.mark.asyncio
@patch(f"{_MOD}.inspect_aawm_alias_routing_durable_key", new_callable=AsyncMock)
@patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
@patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
async def test_inspect_durable_proves_invalidation_after_removal(
    mock_dc,
    mock_inspect,
    mock_key_inspect,
):
    """After the seeded candidate is REMOVED from the active snapshot,
    ``inspect_durable`` re-derives the prior identity from the sanitized
    route_family handle and inspects the prior configured-lane key -- with
    no prepared-run or other global mutable state."""
    mock_dc.return_value = _mock_dual_cache()
    # Prior identity set + key are ABSENT (invalidated).
    mock_inspect.return_value = _mock_identity_inspection(exists=False, cardinality=0)
    mock_key_inspect.return_value = _mock_key_inspection(exists=False, ttl_seconds=0.0)
    request = _make_request()
    assert _prepared_runs == {}

    body = _inspect_prior_body(route_family="codex_openai_responses")
    body["codex_oauth_account_id"] = _CODEX_OAUTH_ACCOUNT_ID
    with patch(
        f"{_MOD}.get_active_routing_snapshot",
        return_value=_removed_snapshot(),
    ):
        result = await _handle_inspect_durable(body, _RUN_ID, request)

    expected_lane = _resolve_cfg019_openai_lane_key(
        request,
        {"provider": _TARGET_PROVIDER, "model": _TARGET_MODEL},
        codex_oauth_account_id=_CODEX_OAUTH_ACCOUNT_ID,
    )
    assert expected_lane == _CONFIGURED_OAUTH_LANE
    expected_key = ca_mod._resolve_cfg019_openai_cooldown_key(
        request,
        {
            "provider": _TARGET_PROVIDER,
            "model": _TARGET_MODEL,
            "codex_oauth_lane_key": expected_lane,
        },
        codex_oauth_account_id=_CODEX_OAUTH_ACCOUNT_ID,
        cooldown_identity_candidate={
            "provider": _TARGET_PROVIDER,
            "model": _TARGET_MODEL,
            "route_family": "codex_openai_responses",
            "cooldown_identity_tag": (
                f"alias:basic:{_TARGET_PROVIDER}:{_TARGET_MODEL}:"
                "codex_openai_responses"
            ),
        },
    )
    assert mock_key_inspect.await_args.kwargs["state_key"] == expected_key
    assert expected_lane in expected_key
    assert "auth:" not in expected_key
    assert result["result"] == "inspected_durable"
    assert result["durable_cooldown_active"] is False
    assert result["state_source"] == "absent"
    assert result["prepared_run_required"] is False
    _assert_sanitized_public_fields(result)
    assert _prepared_runs == {}


@pytest.mark.asyncio
@patch(f"{_MOD}.inspect_aawm_alias_routing_durable_key", new_callable=AsyncMock)
@patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
@patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
async def test_inspect_durable_proves_invalidation_after_semantic_change(
    mock_dc,
    mock_inspect,
    mock_key_inspect,
):
    """After a genuine candidate SEMANTIC change (route_family rotated), the
    prior seeded identity differs from the new one, and ``inspect_durable``
    re-derives the OLD identity to prove it is absent."""
    mock_dc.return_value = _mock_dual_cache()
    mock_inspect.return_value = _mock_identity_inspection(exists=False, cardinality=0)
    mock_key_inspect.return_value = _mock_key_inspection(exists=False, ttl_seconds=0.0)
    request = _make_request()
    assert _prepared_runs == {}

    with patch(f"{_MOD}.get_active_routing_snapshot", return_value=_changed_snapshot()):
        body = _inspect_prior_body(route_family="codex_openai_responses")
        result = await _handle_inspect_durable(body, _RUN_ID, request)

    assert result["result"] == "inspected_durable"
    assert result["durable_cooldown_active"] is False
    assert result["state_source"] == "absent"
    # The re-derived prior identity differs from the active (changed) one.
    active_identity = ca_mod.resolve_lane_identity_hash(
        candidate={
            "cooldown_identity_tag": (
                f"alias:basic:{_TARGET_PROVIDER}:{_TARGET_MODEL}:"
                "codex_openai_chat_completions_adapter"
            )
        }
    )
    assert result["identity_hash"] != ca_mod._sanitize_identifier(active_identity)
    _assert_sanitized_public_fields(result)
    assert _prepared_runs == {}


@pytest.mark.asyncio
@patch(f"{_MOD}.inspect_aawm_alias_routing_durable_key", new_callable=AsyncMock)
@patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
@patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
async def test_inspect_durable_prior_handle_matches_active_candidate(
    mock_dc,
    mock_inspect,
    mock_key_inspect,
):
    """When the prior route_family still matches the active candidate, the
    prior-identity handle resolves to the SAME identity as the active
    candidate (no false invalidation)."""
    mock_dc.return_value = _mock_dual_cache()
    mock_inspect.return_value = _mock_identity_inspection(exists=True, cardinality=1)
    mock_key_inspect.return_value = _mock_key_inspection(ttl_seconds=88.0)
    request = _make_request()

    with patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot()):
        result = await _handle_inspect_durable(
            _inspect_prior_body(route_family="codex_openai_responses"),
            _RUN_ID,
            request,
        )

    assert result["result"] == "inspected_durable"
    assert result["durable_cooldown_active"] is True
    assert result["state_source"] == "durable_cache"
    assert result["ttl_seconds"] == 88.0
    _assert_sanitized_public_fields(result)


# ---------------------------------------------------------------------------
# Finding 3: retain the publish receipt/journal; roll back ONLY that
# transaction on bounded post-seed TTL verification failure.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch(f"{_MOD}.rollback_cooldown_transaction", new_callable=AsyncMock)
@patch(f"{_MOD}.inspect_aawm_alias_routing_durable_key", new_callable=AsyncMock)
@patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
@patch(f"{_MOD}.publish_cooldown_transaction", new_callable=AsyncMock)
@patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
@patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
async def test_seed_durable_rolls_back_only_own_transaction_on_ttl_failure(
    _snap,
    mock_dc,
    mock_publish,
    mock_inspect,
    mock_key_inspect,
    mock_rollback,
):
    """When bounded post-seed TTL verification fails, seed_durable must
    retain the publish receipt/journal and roll back ONLY that transaction
    (not unrelated durable state)."""
    mock_dc.return_value = _mock_dual_cache()
    # Build the exact journal the publish returns; rollback must receive it.
    captured: dict[str, str] = {}

    async def _fake_publish(*, alias_family, identity_hash, cooldown_keys, lane_members, ttl_seconds, **kw):
        captured["identity_hash"] = identity_hash
        captured["cooldown_key"] = cooldown_keys[0]
        return _make_txn_result(
            cooldown_key=cooldown_keys[0], identity_hash=identity_hash
        )

    mock_publish.side_effect = _fake_publish
    mock_inspect.side_effect = [
        _mock_identity_inspection(exists=False, cardinality=0),
        _mock_identity_inspection(exists=True, cardinality=1),
    ]
    # Post-seed key inspection reports an unbounded/persistent TTL -> failure.
    post = _mock_key_inspection(exists=True, ttl_seconds=0.0)
    post.ttl_remaining_seconds = UNBOUNDED_EXPIRY
    post.payload = {"persistent": True}
    mock_key_inspect.side_effect = [
        _mock_key_inspection(exists=False, ttl_seconds=0.0),
        post,
    ]
    state_mgr = AliasRoutingStateManager()
    request = _make_request()

    with pytest.raises(HTTPException) as exc_info:
        await _handle_seed_durable(_seed_body(ttl_seconds=120), _RUN_ID, request, state_mgr)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail["error"] == "seed_verification_failed"
    # The publish receipt/journal was retained and used for rollback.
    mock_publish.assert_awaited_once()
    mock_rollback.assert_awaited_once()
    rollback_kwargs = mock_rollback.await_args.kwargs
    assert rollback_kwargs["alias_family"] == "codex"
    rolled_journal = rollback_kwargs["journal"]
    # Only the seeded transaction's own journal is rolled back.
    assert rolled_journal.cooldown_keys == [captured["cooldown_key"]]
    assert rolled_journal.identity_hash == captured["identity_hash"]
    # No prepared-run or local mutation leaked.
    assert _RUN_ID not in _prepared_runs
    assert state_mgr.codex.cooldown_until_monotonic_by_key == {}
    assert len(state_mgr.lane_identity_index) == 0


@pytest.mark.asyncio
@patch(f"{_MOD}.rollback_cooldown_transaction", new_callable=AsyncMock)
@patch(f"{_MOD}.inspect_aawm_alias_routing_durable_key", new_callable=AsyncMock)
@patch(f"{_MOD}.inspect_identity_set", new_callable=AsyncMock)
@patch(f"{_MOD}.publish_cooldown_transaction", new_callable=AsyncMock)
@patch(f"{_MOD}.get_aawm_alias_routing_dual_cache")
@patch(f"{_MOD}.get_active_routing_snapshot", return_value=_make_snapshot())
async def test_seed_durable_does_not_rollback_on_success(
    _snap,
    mock_dc,
    mock_publish,
    mock_inspect,
    mock_key_inspect,
    mock_rollback,
):
    """On successful bounded verification, no rollback occurs (receipt is
    retained by the publish path for operator inspection, not rolled back)."""
    mock_dc.return_value = _mock_dual_cache()

    async def _fake_publish(*, alias_family, identity_hash, cooldown_keys, lane_members, ttl_seconds, **kw):
        return _make_txn_result(
            cooldown_key=cooldown_keys[0], identity_hash=identity_hash
        )

    mock_publish.side_effect = _fake_publish
    mock_inspect.side_effect = [
        _mock_identity_inspection(exists=False, cardinality=0),
        _mock_identity_inspection(exists=True, cardinality=1),
    ]
    mock_key_inspect.side_effect = [
        _mock_key_inspection(exists=False, ttl_seconds=0.0),
        _mock_key_inspection(ttl_seconds=119.0),
    ]
    state_mgr = AliasRoutingStateManager()
    request = _make_request()

    result = await _handle_seed_durable(_seed_body(ttl_seconds=120), _RUN_ID, request, state_mgr)

    assert result["result"] == "seeded_durable"
    assert result["durable_cooldown_active"] is True
    mock_publish.assert_awaited_once()
    mock_rollback.assert_not_called()
