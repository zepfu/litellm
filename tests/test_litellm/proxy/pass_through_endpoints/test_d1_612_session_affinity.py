"""D1-612 boundary tests: tokenized pre-egress session ownership."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import durable as durable_mod
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import session_affinity as sa


class _FakeRedisClient:
    def __init__(self, parent: "_FakeRedisCache") -> None:
        self._parent = parent

    async def get(self, name: str) -> Any:
        return self._parent._data.get(name)

    async def set(self, name: str, value: Any, nx: bool = False, ex: Any = None) -> bool:
        if nx and name in self._parent._data:
            return False
        if isinstance(value, (bytes, bytearray)):
            encoded = bytes(value)
        elif isinstance(value, str):
            encoded = value.encode("utf-8")
        else:
            encoded = json.dumps(value).encode("utf-8")
        self._parent._data[name] = encoded
        if ex is not None:
            self._parent._ttl[name] = float(ex)
        return True

    async def delete(self, *names: str) -> int:
        deleted = 0
        for name in names:
            if name in self._parent._data:
                self._parent._data.pop(name, None)
                self._parent._ttl.pop(name, None)
                deleted += 1
        return deleted

    async def persist(self, name: str) -> bool:
        self._parent._ttl.pop(name, None)
        return True

    async def eval(self, script: str, numkeys: int, *args: Any) -> Any:
        # Minimal Lua semantics used by session_affinity promote/release/renew.
        key = args[0]
        raw = self._parent._data.get(key)
        if "PERSIST" in script and "reservation_token" in script and "owned" in script:
            # promote
            token = args[1]
            payload_json = args[2]
            if raw is None:
                return [0, "missing"]
            current = json.loads(raw.decode("utf-8"))
            if current.get("state") == "owned":
                return [2, json.dumps(current)]
            if current.get("state") != "reserved" or current.get("reservation_token") != token:
                return [0, json.dumps(current)]
            payload = json.loads(payload_json)
            if current.get("reserved_at_epoch") is not None:
                payload["reserved_at_epoch"] = current["reserved_at_epoch"]
            self._parent._data[key] = json.dumps(payload).encode("utf-8")
            self._parent._ttl.pop(key, None)
            return [1, json.dumps(payload)]
        if "DEL" in script and "reserved" in script:
            token = args[1]
            if raw is None:
                return 0
            current = json.loads(raw.decode("utf-8"))
            if current.get("state") == "owned":
                return 2
            if current.get("state") == "reserved" and current.get("reservation_token") == token:
                self._parent._data.pop(key, None)
                self._parent._ttl.pop(key, None)
                return 1
            return 0
        if "last_renewed_at_epoch" in script or ("EX" in script and "reserved" in script):
            token = args[1]
            payload_json = args[2]
            ttl = float(args[3])
            if raw is None:
                return 0
            current = json.loads(raw.decode("utf-8"))
            if current.get("state") != "reserved" or current.get("reservation_token") != token:
                return 0
            payload = json.loads(payload_json)
            self._parent._data[key] = json.dumps(payload).encode("utf-8")
            self._parent._ttl[key] = ttl
            return 1
        raise AssertionError(f"unexpected eval script: {script[:80]}")


class _FakeRedisCache:
    def __init__(self, namespace: str = "litellm") -> None:
        self.namespace = namespace
        self._data: dict[str, bytes] = {}
        self._ttl: dict[str, float] = {}
        self.set_calls: list[dict[str, Any]] = []
        self.fail_set_error: Optional[BaseException] = None
        self._client = _FakeRedisClient(self)

    def check_and_fix_namespace(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def init_async_client(self) -> _FakeRedisClient:
        return self._client

    def _get_cache_logic(self, cached_response: Any) -> Any:
        if cached_response is None:
            return None
        if isinstance(cached_response, (bytes, bytearray)):
            cached_response = cached_response.decode("utf-8")
        if isinstance(cached_response, str):
            return json.loads(cached_response)
        return cached_response

    async def async_set_cache(self, key: str, value: Any, **kwargs: Any) -> Any:
        raise_on_error = bool(kwargs.pop("raise_on_error", False))
        try:
            if self.fail_set_error is not None:
                raise self.fail_set_error
            namespaced = self.check_and_fix_namespace(key=key)
            ttl = kwargs.get("ttl")
            nx = bool(kwargs.get("nx", False))
            self.set_calls.append(
                {"key": key, "namespaced_key": namespaced, "value": value, "ttl": ttl, "nx": nx}
            )
            return await self._client.set(
                name=namespaced,
                value=json.dumps(value),
                nx=nx,
                ex=ttl,
            )
        except Exception:
            if raise_on_error:
                raise
            return None


class _FakeDualCache:
    def __init__(self, redis_cache: Optional[_FakeRedisCache]) -> None:
        self.redis_cache = redis_cache


def _patch_dual(redis_cache: Optional[_FakeRedisCache]):
    dual = None if redis_cache is None else _FakeDualCache(redis_cache)
    return patch.object(
        durable_mod, "get_aawm_alias_routing_dual_cache", return_value=dual
    )


def _full_attrs(**over: Any) -> dict[str, Any]:
    base = {
        "provider": "openai",
        "model": "gpt-5.6-sol",
        "route_family": "codex_oauth",
        "endpoint_contract": "openai_responses",
        "state_format": "openai_responses",
        "account_hash": "acct-1",
        "account_label": "primary",
        "account_lane": "lane-a",
    }
    base.update(over)
    return base


@pytest.mark.asyncio
async def test_reservation_race_only_one_competitor_reserves() -> None:
    redis = _FakeRedisCache()
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        attrs_a = _full_attrs(model="model-a")
        attrs_b = _full_attrs(model="model-b")
        g1, g2 = await asyncio.gather(
            sa.guard_session_owner_before_egress(
                session_identity="sess-race",
                requested_attributes=attrs_a,
            ),
            sa.guard_session_owner_before_egress(
                session_identity="sess-race",
                requested_attributes=attrs_b,
            ),
        )
    winners = [
        g
        for g in (g1, g2)
        if g.decision is sa.SessionOwnerGuardDecision.UNOWNED_RESERVED
    ]
    losers = [
        g
        for g in (g1, g2)
        if g.decision is sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED
    ]
    assert len(winners) == 1
    assert len(losers) == 1
    assert winners[0].held_reservation is True
    assert losers[0].held_reservation is False
    assert "concurrent" in (losers[0].mismatch_reason or "")


@pytest.mark.asyncio
async def test_streaming_pre_send_requires_reservation_before_upstream() -> None:
    """pass_through_request pre-send guard reserves before send."""
    redis = _FakeRedisCache()
    order: list[str] = []

    async def fake_guard(**kwargs: Any):
        order.append("guard")
        return await real_ensure(**kwargs)

    real_ensure = sa.ensure_session_owner_guard_for_request
    from types import SimpleNamespace

    request = MagicMock()
    request.state = SimpleNamespace()
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ), patch.object(sa, "ensure_session_owner_guard_for_request", side_effect=fake_guard):
        # call internal helper used immediately before upstream wait
        from litellm.proxy.pass_through_endpoints import pass_through_endpoints as pte

        await pte._aawm_session_owner_pre_send_guard(
            request=request,
            parsed_body={
                "model": "gpt-test",
                "litellm_metadata": {"session_id": "sess-stream"},
            },
            custom_llm_provider="openai",
            egress_credential_family="openai",
            expected_target_family="openai",
            url=None,
        )
        order.append("upstream_send_would_start")
    assert order == ["guard", "upstream_send_would_start"]
    assert request.state._aawm_session_owner_guarded is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route_attrs",
    [
        _full_attrs(
            provider="openai",
            model="gpt-5.6-sol",
            route_family="codex_oauth",
            endpoint_contract="openai_responses",
            state_format="openai_responses",
        ),
        _full_attrs(
            provider="anthropic",
            model="claude-sonnet",
            route_family="anthropic_native",
            endpoint_contract="anthropic_messages",
            state_format="anthropic",
            account_hash=None,
            account_label=None,
            account_lane=None,
        ),
        _full_attrs(
            provider="openai",
            model="gpt-direct",
            route_family="openai",
            endpoint_contract="openai_passthrough",
            state_format="openai",
            account_hash=None,
            account_label=None,
            account_lane=None,
        ),
        _full_attrs(
            provider="xai",
            model="grok-4.5",
            route_family="codex_nested",
            endpoint_contract="openai_responses",
            state_format="openai_responses",
            account_hash=None,
            account_label=None,
            account_lane=None,
        ),
        _full_attrs(
            provider="anthropic",
            model="claude-nested",
            route_family="anthropic_nested",
            endpoint_contract="anthropic_messages",
            state_format="anthropic",
            account_hash=None,
            account_label=None,
            account_lane=None,
        ),
    ],
)
async def test_route_family_reserve_promote_lifecycle(route_attrs: dict[str, Any]) -> None:
    redis = _FakeRedisCache()
    # Drop None account fields for non-account routes
    attrs = {k: v for k, v in route_attrs.items() if v is not None}
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        guard = await sa.guard_session_owner_before_egress(
            session_identity="sess-family",
            requested_attributes=attrs,
        )
        assert guard.decision is sa.SessionOwnerGuardDecision.UNOWNED_RESERVED
        promote = await sa.promote_session_owner_reservation(
            session_identity="sess-family",
            reservation_token=guard.reservation_token,
            attributes=attrs,
        )
        assert promote.outcome is sa.SessionOwnerMutationOutcome.PROMOTED
        assert promote.owner_record is not None
        assert promote.owner_record.get("state") == "owned"
        assert promote.owner_record.get("persistent") is True
        # owned retention: second request compatible
        g2 = await sa.guard_session_owner_before_egress(
            session_identity="sess-family",
            requested_attributes=attrs,
        )
        assert g2.decision is sa.SessionOwnerGuardDecision.COMPATIBLE_OWNER


@pytest.mark.asyncio
async def test_account_mismatch_fails_before_egress() -> None:
    redis = _FakeRedisCache()
    attrs = _full_attrs(account_hash="acct-1", account_lane="lane-a")
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        g = await sa.guard_session_owner_before_egress(
            session_identity="sess-acct", requested_attributes=attrs
        )
        await sa.promote_session_owner_reservation(
            session_identity="sess-acct",
            reservation_token=g.reservation_token,
            attributes=attrs,
        )
        bad = dict(attrs)
        bad["account_hash"] = "acct-other"
        g2 = await sa.guard_session_owner_before_egress(
            session_identity="sess-acct", requested_attributes=bad
        )
        assert g2.decision is sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED
        with pytest.raises(HTTPException) as ei:
            sa.raise_session_owner_redispatch_required(
                session_identity="sess-acct", guard=g2
            )
        assert ei.value.status_code == 409
        detail = ei.value.detail
        assert detail["redispatch_required"] is True
        assert detail["attempted_provider_call"] is False


@pytest.mark.asyncio
async def test_redis_failure_fail_closed_before_egress() -> None:
    with _patch_dual(None), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        g = await sa.guard_session_owner_before_egress(
            session_identity="sess-redis-down",
            requested_attributes=_full_attrs(),
        )
    assert g.decision is sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED
    assert "durable cache unavailable" in (g.mismatch_reason or "")


@pytest.mark.asyncio
async def test_owner_retention_and_release_on_failure() -> None:
    redis = _FakeRedisCache()
    attrs = _full_attrs()
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        g = await sa.guard_session_owner_before_egress(
            session_identity="sess-ret", requested_attributes=attrs
        )
        assert g.held_reservation
        # failure releases reservation so another request can take over
        rel = await sa.release_session_owner_reservation(
            session_identity="sess-ret",
            reservation_token=g.reservation_token,
        )
        assert rel.outcome is sa.SessionOwnerMutationOutcome.RELEASED
        g2 = await sa.guard_session_owner_before_egress(
            session_identity="sess-ret", requested_attributes=attrs
        )
        assert g2.decision is sa.SessionOwnerGuardDecision.UNOWNED_RESERVED
        # promote and retain
        await sa.promote_session_owner_reservation(
            session_identity="sess-ret",
            reservation_token=g2.reservation_token,
            attributes=attrs,
        )
        # release must not delete owned
        rel2 = await sa.release_session_owner_reservation(
            session_identity="sess-ret",
            reservation_token=g2.reservation_token,
        )
        assert rel2.outcome is sa.SessionOwnerMutationOutcome.ALREADY_OWNED
        g3 = await sa.guard_session_owner_before_egress(
            session_identity="sess-ret", requested_attributes=attrs
        )
        assert g3.decision is sa.SessionOwnerGuardDecision.COMPATIBLE_OWNER


@pytest.mark.asyncio
async def test_canonical_key_ignores_alias_provider_lane_prefix() -> None:
    with patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        k1 = sa.build_aawm_alias_routing_session_owner_cache_key(
            session_identity="sess-abc"
        )
        k2 = sa.build_aawm_alias_routing_session_owner_cache_key(
            session_identity=sa.resolve_canonical_session_identity(
                session_identity="work:sess-abc:lane-1"
            )
            or ""
        )
    assert k1 == k2
    assert ":session_owner:" in k1


@pytest.mark.asyncio
async def test_account_scoped_incomplete_fails_before_egress_not_after() -> None:
    """Account-scoped routes must not reserve/send without credential identity."""
    redis = _FakeRedisCache()
    incomplete = {
        "provider": "openai",
        "model": "gpt",
        "route_family": "codex_oauth",
        "endpoint_contract": "openai_responses",
        "state_format": "openai_responses",
        # missing account_* for account-scoped route
    }
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        g = await sa.guard_session_owner_before_egress(
            session_identity="sess-incomplete",
            requested_attributes=incomplete,
        )
    assert g.decision is sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED
    assert "account-scoped" in (g.mismatch_reason or "")
    assert g.held_reservation is False
    assert redis.set_calls == []


@pytest.mark.asyncio
async def test_known_owner_attributes_pin_exactly_not_fuzzy() -> None:
    redis = _FakeRedisCache()
    owned = _full_attrs(model="model-a", account_hash="acct-1", account_lane="lane-a")
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        g = await sa.guard_session_owner_before_egress(
            session_identity="sess-exact", requested_attributes=owned
        )
        await sa.promote_session_owner_reservation(
            session_identity="sess-exact",
            reservation_token=g.reservation_token,
            attributes=owned,
        )
        # Same provider family but different model must not pass as "compatible".
        fuzzy = dict(owned)
        fuzzy["model"] = "model-b"
        g2 = await sa.guard_session_owner_before_egress(
            session_identity="sess-exact", requested_attributes=fuzzy
        )
    assert g2.decision is sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED
    assert "exactly match" in (g2.mismatch_reason or "") or "mismatch" in (
        g2.mismatch_reason or ""
    )


@pytest.mark.asyncio
async def test_finalize_request_lease_promotes_stream_object_and_releases_on_failure() -> None:
    """Shared lifecycle: stream-like success object promotes; failure releases."""
    redis = _FakeRedisCache()
    attrs = _full_attrs()
    request = type("Req", (), {})()
    request.state = type("State", (), {})()
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        guard = await sa.ensure_session_owner_guard_for_request(
            request=request,
            session_identity="sess-stream-lease",
            requested_attributes=attrs,
        )
        assert guard.held_reservation is True
        lease = sa.get_request_session_owner_lease(request)
        assert lease is not None and lease.held_reservation

        # Stream-like object without status_code counts as first-byte success.
        stream_obj = object()
        result = await sa.finalize_request_session_owner_lease(
            request, stream_obj, attributes=attrs
        )
        assert result is not None
        assert result.outcome is sa.SessionOwnerMutationOutcome.PROMOTED
        assert lease.promoted is True

        # New reservation for failure path
        request2 = type("Req", (), {})()
        request2.state = type("State", (), {})()
        guard2 = await sa.ensure_session_owner_guard_for_request(
            request=request2,
            session_identity="sess-fail-lease",
            requested_attributes=attrs,
        )
        assert guard2.held_reservation is True
        await sa.finalize_request_session_owner_lease(
            request2, exc=RuntimeError("upstream boom")
        )
        lease2 = sa.get_request_session_owner_lease(request2)
        assert lease2 is not None and lease2.released is True
        # Session free again
        g3 = await sa.guard_session_owner_before_egress(
            session_identity="sess-fail-lease", requested_attributes=attrs
        )
        assert g3.decision is sa.SessionOwnerGuardDecision.UNOWNED_RESERVED


@pytest.mark.asyncio
async def test_anthropic_nested_pre_egress_promotes_concrete_not_generic_attrs() -> None:
    """Nested Anthropic reserve/promote must use concrete resolved owner attrs.

    Generic ``anthropic/<inbound>/anthropic_nested`` placeholders must not be
    the promoted owner identity once the adapter has resolved provider, model,
    route_family, endpoint_contract, state_format, and safe account lane.
    """
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
        anthropic_adapter_calls as aac,
    )

    redis = _FakeRedisCache()
    request = type("Req", (), {})()
    request.state = type("State", (), {})()
    inbound_model = "claude-sonnet-inbound-alias"
    body = {
        "model": inbound_model,
        "litellm_metadata": {
            "account_lane": "lane-nested-a",
            "account_label": "nested-primary",
            "account_hash": "nested-hash-1",
        },
    }
    concrete_provider = "xai"
    concrete_model = "grok-4.5"
    concrete_route = "anthropic_xai_oauth_completion_adapter"

    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ), patch.object(
        sa, "resolve_canonical_session_identity", return_value="sess-nested-concrete"
    ):
        # Last common nested pre-egress point after concrete resolution.
        await aac._ensure_anthropic_nested_session_owner_pre_egress(
            request=request,
            request_body=body,
            provider=concrete_provider,
            model=concrete_model,
            route_family=concrete_route,
            endpoint_contract="anthropic_messages",
            state_format="anthropic",
            failure_phase="session_owner_anthropic_nested_pre_egress",
        )
        lease = sa.get_request_session_owner_lease(request)
        assert lease is not None
        assert lease.held_reservation is True
        assert lease.attributes is not None
        assert lease.attributes.get("provider") == concrete_provider
        assert lease.attributes.get("model") == concrete_model
        assert lease.attributes.get("route_family") == concrete_route
        assert lease.attributes.get("endpoint_contract") == "anthropic_messages"
        assert lease.attributes.get("state_format") == "anthropic"
        assert lease.attributes.get("account_lane") == "lane-nested-a"
        # Must not promote the generic inbound/nested placeholder identity.
        assert lease.attributes.get("route_family") != "anthropic_nested"
        assert lease.attributes.get("provider") != "anthropic"
        assert lease.attributes.get("model") != inbound_model

        result = await sa.finalize_request_session_owner_lease(
            request,
            response=type("R", (), {"status_code": 200})(),
            failure_phase="session_owner_anthropic_nested_promote",
        )
        assert result is not None
        assert result.outcome is sa.SessionOwnerMutationOutcome.PROMOTED
        owner = result.owner_record or {}
        assert owner.get("state") == "owned"
        owner_attrs = owner.get("attributes") or {}
        assert owner_attrs.get("provider") == concrete_provider
        assert owner_attrs.get("model") == concrete_model
        assert owner_attrs.get("route_family") == concrete_route
        assert owner_attrs.get("endpoint_contract") == "anthropic_messages"
        assert owner_attrs.get("state_format") == "anthropic"
        assert owner_attrs.get("account_lane") == "lane-nested-a"
        assert owner_attrs.get("route_family") != "anthropic_nested"
        assert owner_attrs.get("provider") != "anthropic"
        assert owner_attrs.get("model") != inbound_model

        # Double finalization is a no-op (no second promote/release).
        again = await sa.finalize_request_session_owner_lease(
            request,
            response=type("R", (), {"status_code": 200})(),
        )
        assert again is None


@pytest.mark.asyncio
async def test_codex_nested_concrete_same_owner_continuation_passes() -> None:
    """Valid nested Codex same-owner continuation must pass exact pre-egress guard.

    A prior successful turn (or alias candidate_loop) may already own/reserve a
    concrete provider/model/route. Nested Codex pre-egress with the same
    concrete attributes must continue without redispatch and must not treat
    generic ``codex_nested`` as the owner identity.
    """
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
        codex_dispatch as cd,
    )

    redis = _FakeRedisCache()
    request = type("Req", (), {})()
    request.state = type("State", (), {})()
    sid = "sess-codex-same-owner"
    concrete = {
        "provider": "kimi_code",
        "model": "kimi-k2.5",
        "route_family": "codex_kimi_chat_completions_adapter",
        "endpoint_contract": "openai_responses",
        "state_format": "openai_responses",
        "account_lane": "lane-codex-a",
        "account_label": "codex-primary",
        "account_hash": "codex-hash-1",
    }
    body = {
        "model": "inbound-codex-alias",
        "litellm_metadata": {
            "account_lane": concrete["account_lane"],
            "account_label": concrete["account_label"],
            "account_hash": concrete["account_hash"],
        },
    }

    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ), patch.object(sa, "resolve_canonical_session_identity", return_value=sid):
        # Existing concrete owner established by a prior successful turn.
        g0 = await sa.guard_session_owner_before_egress(
            session_identity=sid, requested_attributes=concrete
        )
        assert g0.held_reservation is True
        promo = await sa.promote_session_owner_reservation(
            session_identity=sid,
            reservation_token=g0.reservation_token,
            attributes=concrete,
        )
        assert promo.outcome is sa.SessionOwnerMutationOutcome.PROMOTED

        # Early consult must not reject on generic codex_nested placeholders.
        await cd._consult_codex_nested_session_owner(
            request=request, prepared_request_body=body
        )
        affinity = getattr(request.state, "_aawm_session_owner_consult_affinity", None)
        assert isinstance(affinity, dict)
        assert affinity.get("provider") == concrete["provider"]
        assert affinity.get("model") == concrete["model"]

        # Simulate candidate_loop holding a concrete same-owner reservation.
        cont = await sa.ensure_session_owner_guard_for_request(
            request=request,
            request_body=body,
            session_identity=sid,
            requested_attributes=concrete,
            require_exact_attributes=True,
        )
        assert cont.decision is sa.SessionOwnerGuardDecision.COMPATIBLE_OWNER

        # Nested Codex pre-egress with the same concrete attrs must pass.
        await cd._ensure_codex_nested_session_owner_pre_egress(
            request=request,
            request_body=body,
            session_identity=sid,
            provider=concrete["provider"],
            model=concrete["model"],
            route_family=concrete["route_family"],
        )
        lease = sa.get_request_session_owner_lease(request)
        assert lease is not None
        attrs = lease.attributes or {}
        assert attrs.get("provider") == concrete["provider"]
        assert attrs.get("model") == concrete["model"]
        assert attrs.get("route_family") == concrete["route_family"]
        assert attrs.get("account_lane") == concrete["account_lane"]
        assert attrs.get("route_family") != "codex_nested"
        assert attrs.get("provider") != "openai"


@pytest.mark.asyncio
async def test_codex_nested_concrete_mismatch_fails_before_send() -> None:
    """Concrete nested Codex mismatch must raise redispatch_required pre-send."""
    from fastapi import HTTPException

    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
        codex_dispatch as cd,
    )

    redis = _FakeRedisCache()
    request = type("Req", (), {})()
    request.state = type("State", (), {})()
    sid = "sess-codex-mismatch"
    owned = {
        "provider": "kimi_code",
        "model": "kimi-k2.5",
        "route_family": "codex_kimi_chat_completions_adapter",
        "endpoint_contract": "openai_responses",
        "state_format": "openai_responses",
        "account_lane": "lane-codex-a",
        "account_label": "codex-primary",
        "account_hash": "codex-hash-1",
    }
    mismatched = {
        "provider": "alibaba_token_plan",
        "model": "qwen3-coder-plus",
        "route_family": "codex_alibaba_token_plan_chat_completions_adapter",
    }
    body = {
        "model": "inbound-codex-alias",
        "litellm_metadata": {
            "account_lane": owned["account_lane"],
            "account_label": owned["account_label"],
            "account_hash": owned["account_hash"],
        },
    }

    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ), patch.object(sa, "resolve_canonical_session_identity", return_value=sid):
        g0 = await sa.guard_session_owner_before_egress(
            session_identity=sid, requested_attributes=owned
        )
        await sa.promote_session_owner_reservation(
            session_identity=sid,
            reservation_token=g0.reservation_token,
            attributes=owned,
        )

        # Early consult must NOT reject solely because generic codex_nested
        # would mismatch the concrete owner.
        await cd._consult_codex_nested_session_owner(
            request=request, prepared_request_body=body
        )
        assert getattr(request.state, "_aawm_session_owner_consult_affinity", None)

        # Concrete mismatched nested route must fail closed before provider I/O.
        with pytest.raises(HTTPException) as ei:
            await cd._ensure_codex_nested_session_owner_pre_egress(
                request=request,
                request_body=body,
                session_identity=sid,
                provider=mismatched["provider"],
                model=mismatched["model"],
                route_family=mismatched["route_family"],
            )
        detail = ei.value.detail
        assert isinstance(detail, dict)
        assert detail.get("redispatch_required") is True
        # No provider send occurred; request must not hold a successful
        # mismatched reservation/promotion.
        lease = sa.get_request_session_owner_lease(request)
        if lease is not None:
            assert lease.promoted is not True
            assert (lease.attributes or {}).get("provider") != mismatched["provider"] or (
                lease.held_reservation is not True and lease.promoted is not True
            )


@pytest.mark.asyncio
async def test_extract_chatgpt_account_id_uses_safe_hash_not_raw() -> None:
    """Inbound chatgpt-account-id becomes one-way hash/lane; raw never returned."""
    raw_id = "acct_RAW_SECRET_VALUE_9f3c"
    digest = sa._hash_account_identity_value(raw_id)
    request = type("Req", (), {})()
    request.headers = {"chatgpt-account-id": raw_id, "authorization": "Bearer secret-token"}

    identity = sa.extract_account_identity_from_context(request=request)
    assert identity.get("account_hash") == f"chatgpt-account-hash:{digest}"
    assert identity.get("account_lane") == f"chatgpt-account:{digest}"
    # No field may contain the raw account id or bearer secret.
    blob = json.dumps(identity, sort_keys=True)
    assert raw_id not in blob
    assert "secret-token" not in blob
    assert "Bearer" not in blob
    assert "chatgpt-account-id" not in blob

    # Header alias / body alias also hashed; never passthrough.
    identity2 = sa.extract_account_identity_from_context(
        headers={"ChatGPT-Account-Id": raw_id}
    )
    assert identity2.get("account_hash") == f"chatgpt-account-hash:{digest}"
    assert raw_id not in json.dumps(identity2)

    # Pre-existing safe oauth metadata wins over header-derived values.
    identity3 = sa.extract_account_identity_from_context(
        request=request,
        request_body={
            "litellm_metadata": {
                "codex_oauth_account_hash": "preexisting-hash",
                "codex_oauth_lane_key": "codex-oauth:primary:preexisting-hash",
                "codex_oauth_account_label": "primary",
            }
        },
    )
    assert identity3.get("account_hash") == "preexisting-hash"
    assert identity3.get("account_lane") == "codex-oauth:primary:preexisting-hash"
    assert identity3.get("account_label") == "primary"
    assert raw_id not in json.dumps(identity3)


@pytest.mark.asyncio
async def test_direct_openai_pre_egress_uses_concrete_attrs_and_safe_account() -> None:
    """Direct OpenAI/Codex fallthrough owner attrs are concrete + safe account."""
    redis = _FakeRedisCache()
    raw_id = "acct_DIRECT_RAW_77"
    digest = sa._hash_account_identity_value(raw_id)
    request = type("Req", (), {})()
    request.state = type("State", (), {})()
    request.headers = {
        "chatgpt-account-id": raw_id,
        "originator": "codex_cli_rs",
        "session_id": "sess-direct-1",
    }
    body = {"model": "gpt-5.4-codex"}
    sid = "sess-direct-concrete"

    # Mirror the direct-handler attribute construction with resolved concrete
    # provider/model/route and safe account extraction.
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ), patch.object(sa, "resolve_canonical_session_identity", return_value=sid):
        account_identity = sa.extract_account_identity_from_context(
            request=request, request_body=body
        )
        attrs = sa.build_session_owner_attributes(
            provider="openai",
            model="gpt-5.4-codex",
            route_family="codex_oauth",
            endpoint_contract="openai_responses",
            state_format="openai_responses",
            ingress="openai_passthrough",
            requested_model="gpt-5.4-codex",
            extra=account_identity,
        )
        assert attrs.get("provider") == "openai"
        assert attrs.get("model") == "gpt-5.4-codex"
        assert attrs.get("route_family") == "codex_oauth"
        assert attrs.get("endpoint_contract") == "openai_responses"
        assert attrs.get("state_format") == "openai_responses"
        assert attrs.get("account_hash") == f"chatgpt-account-hash:{digest}"
        assert attrs.get("account_lane") == f"chatgpt-account:{digest}"
        assert attrs.get("route_family") != "openai" or attrs.get("model") == "gpt-5.4-codex"
        assert raw_id not in json.dumps(attrs)

        guard = await sa.ensure_session_owner_guard_for_request(
            request=request,
            request_body=body,
            session_identity=sid,
            requested_attributes=attrs,
            require_exact_attributes=True,
            failure_phase="session_owner_direct_openai_pre_egress",
        )
        assert guard.held_reservation is True
        lease = sa.get_request_session_owner_lease(request)
        assert lease is not None
        assert lease.attributes.get("model") == "gpt-5.4-codex"
        assert lease.attributes.get("route_family") == "codex_oauth"
        assert lease.attributes.get("account_lane") == f"chatgpt-account:{digest}"
        assert raw_id not in json.dumps(lease.attributes or {})


@pytest.mark.asyncio
async def test_direct_account_scoped_missing_identity_fails_before_send() -> None:
    """Account-scoped direct Codex/OpenAI route fails closed without account id."""
    redis = _FakeRedisCache()
    # codex_oauth is account-scoped; no chatgpt-account-id and no safe metadata.
    incomplete = sa.build_session_owner_attributes(
        provider="openai",
        model="gpt-5.4-codex",
        route_family="codex_oauth",
        endpoint_contract="openai_responses",
        state_format="openai_responses",
        ingress="openai_passthrough",
        requested_model="gpt-5.4-codex",
        extra=sa.extract_account_identity_from_context(
            request=type("Req", (), {"headers": {}})(),
            request_body={"model": "gpt-5.4-codex"},
        ),
    )
    assert sa.route_requires_account_identity(incomplete) is True
    reason = sa.incomplete_owner_attribute_reason(incomplete, for_promotion=True)
    assert reason is not None
    assert "account-scoped" in reason

    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        g = await sa.guard_session_owner_before_egress(
            session_identity="sess-direct-missing-acct",
            requested_attributes=incomplete,
        )
    assert g.decision is sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED
    assert g.held_reservation is False
    assert redis.set_calls == []
    assert "account-scoped" in (g.mismatch_reason or "")

    # Non-account-scoped direct OpenAI passthrough remains valid without account.
    non_account = sa.build_session_owner_attributes(
        provider="openai",
        model="gpt-4o-mini",
        route_family="openai",
        endpoint_contract="openai_passthrough",
        state_format="openai",
        ingress="openai_passthrough",
        requested_model="gpt-4o-mini",
    )
    assert sa.route_requires_account_identity(non_account) is False
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        g2 = await sa.guard_session_owner_before_egress(
            session_identity="sess-direct-non-account",
            requested_attributes=non_account,
        )
    assert g2.decision is sa.SessionOwnerGuardDecision.UNOWNED_RESERVED
    assert g2.held_reservation is True


@pytest.mark.asyncio
async def test_selection_helpers_published_on_host_install() -> None:
    """D1-612 helpers must install through _HOST_FUNCTION_NAMES rebinding."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import selection as sel

    assert "_session_affinity_mod" in sel._HOST_FUNCTION_NAMES
    assert "_attach_session_owner_selection_fields" in sel._HOST_FUNCTION_NAMES
    host: dict = {}
    # Minimal host env for install side effects; may pull many symbols.
    # Call install with a dict that already has required names from module.
    host.update({k: getattr(sel, k) for k in dir(sel) if not k.startswith("__")})
    sel.install(host)
    assert callable(host.get("_session_affinity_mod"))
    assert callable(host.get("_attach_session_owner_selection_fields"))
    # Rebound objects must be the installed host copies (no NameError path).
    assert host["_session_affinity_mod"] is sel._session_affinity_mod
    assert host["_attach_session_owner_selection_fields"] is (
        sel._attach_session_owner_selection_fields
    )
    # Helper remains usable after rebind.
    mod = host["_session_affinity_mod"]()
    assert hasattr(mod, "guard_session_owner_before_egress")


@pytest.mark.asyncio
async def test_no_fixed_six_hour_owned_expiry() -> None:
    redis = _FakeRedisCache()
    attrs = _full_attrs()
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        g = await sa.guard_session_owner_before_egress(
            session_identity="sess-persist", requested_attributes=attrs
        )
        # reservation uses short ttl
        assert any(
            c.get("nx") and c.get("ttl") and float(c["ttl"]) <= 900
            for c in redis.set_calls
        )
        p = await sa.promote_session_owner_reservation(
            session_identity="sess-persist",
            reservation_token=g.reservation_token,
            attributes=attrs,
        )
    assert p.owner_record is not None
    assert p.owner_record.get("persistent") is True
    assert "expires_at_epoch" not in (p.owner_record or {})
