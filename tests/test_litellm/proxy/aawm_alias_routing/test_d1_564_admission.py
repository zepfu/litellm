"""Focused tests for D1-564 provider-lane admission (first durable body)."""

from __future__ import annotations

import asyncio
import inspect
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request
from starlette.responses import Response

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import admission
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.candidate_loop import (
    handle_alias_route,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.interfaces import (
    AliasRouteServices,
    CooldownPublicationPlan,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    alias_routing_state,
)


@pytest.fixture(autouse=True)
def _reset_admission_state():
    admission.reset_admission_state_for_tests()
    alias_routing_state.reset_for_tests()
    yield
    admission.reset_admission_state_for_tests()
    alias_routing_state.reset_for_tests()


def _candidate(**overrides: Any) -> dict[str, Any]:
    base = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "route_family": "anthropic_messages",
        "last_resort": False,
        "codex_oauth_account_hash": "acctdeadbeef12",
        "codex_oauth_account_label": "account1",
        "codex_oauth_lane_key": "anthropic:account1:acctdeadbeef12",
    }
    base.update(overrides)
    return base


def _selection(candidate: Optional[dict[str, Any]] = None, **overrides: Any) -> dict[str, Any]:
    cand = candidate or _candidate()
    base = {
        "candidate": cand,
        "lane_key": cand.get("codex_oauth_lane_key") or "anthropic",
        "cooldown_key": "anthropic:claude-sonnet-4-5:anthropic",
        "session_key": "session-1",
        "selection_reason": "first_available",
        "skipped": [],
        "in_flight_session": False,
        "cooldown_seconds": 0.0,
        "cooldown_state_source": "none",
    }
    base.update(overrides)
    return base


def _minimal_request(session_id: str = "admission-test") -> MagicMock:
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.headers = {"session_id": session_id}
    request.query_params = {}
    request.url = MagicMock()
    request.scope = {
        "path": "/v1/messages",
        "query_string": b"",
        "parsed_body": None,
    }
    request.state = SimpleNamespace(
        aawm_alias_request_local_cooldown_until={},
        aawm_alias_request_local_excluded_keys=set(),
    )
    return request


def _plan() -> CooldownPublicationPlan:
    return CooldownPublicationPlan(
        memory_keys=("k",),
        durable_keys=("k",),
        duration_seconds=1.0,
        applied_scope="model",
        request_local_action="none",
    )


def test_lane_fingerprint_is_provider_account_not_alias_or_session() -> None:
    fp_a = admission.build_provider_account_lane_fingerprint(
        provider="anthropic",
        account_hash="acctdeadbeef12",
    )
    fp_b = admission.build_provider_account_lane_fingerprint(
        provider="anthropic",
        account_hash="acctdeadbeef12",
        lane_key="ignored-for-account-lanes",
    )
    fp_other = admission.build_provider_account_lane_fingerprint(
        provider="anthropic",
        account_hash="otheraccount99",
    )
    assert fp_a == fp_b
    assert fp_a != fp_other
    assert "acctdeadbeef12" not in fp_a
    assert "session" not in fp_a
    # Cache keys stay in the admission keyspace, not cooldown/session-owner.
    counter = admission.build_admission_counter_cache_key(lane_fingerprint=fp_a)
    assert ":admission:" in counter
    assert "cooldown" not in counter
    assert "session-owner" not in counter and "session_owner" not in counter


def test_confirmed_exhaustion_requires_current_usage_reset() -> None:
    now = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
    future = now + timedelta(hours=5)
    past = now - timedelta(minutes=1)
    good = {
        "exhausted": True,
        "remaining_pct": 0.0,
        "status": "fresh",
        "quota_period": "five_hour",
        "window_minutes": 300,
        "limit_scope": "unified",
        "provider_resets_at": future,
        "exhaustion_kind": "usage_limit_reached",
    }
    assert admission.is_confirmed_account_usage_exhaustion(
        good, now_epoch=now.timestamp()
    )
    stale = dict(good, provider_resets_at=past)
    assert not admission.is_confirmed_account_usage_exhaustion(
        stale, now_epoch=now.timestamp()
    )
    projected = {
        "exhausted": False,
        "remaining_pct": 12.0,
        "status": "fresh",
        "limit_scope": "unknown",
    }
    assert not admission.is_confirmed_account_usage_exhaustion(projected)


@pytest.mark.asyncio
async def test_reserve_and_release_local_atomic_capacity_seam() -> None:
    candidate = _candidate()
    selection = _selection(candidate)
    first = await admission.reserve_provider_lane_admission(
        candidate=candidate,
        selection=selection,
        max_in_flight=1,
        lease_ttl_seconds=30,
    )
    assert first.allowed is True
    assert first.lease is not None
    second = await admission.reserve_provider_lane_admission(
        candidate=candidate,
        selection=selection,
        max_in_flight=1,
        lease_ttl_seconds=30,
    )
    assert second.allowed is False
    assert second.reason == admission.AdmissionDenyReason.CAPACITY_UNAVAILABLE.value
    assert second.limit_scope == "concurrency"
    released = await admission.release_provider_lane_admission(first.lease)
    assert released is True
    third = await admission.reserve_provider_lane_admission(
        candidate=candidate,
        selection=selection,
        max_in_flight=1,
        lease_ttl_seconds=30,
    )
    assert third.allowed is True
    await admission.release_provider_lane_admission(third.lease)


@pytest.mark.asyncio
async def test_confirmed_exhaustion_returns_exact_reset_structured_429() -> None:
    reset_at = datetime(2026, 8, 11, 18, 0, tzinfo=timezone.utc)
    observation = {
        "exhausted": True,
        "remaining_pct": 0.0,
        "status": "fresh",
        "quota_period": "five_hour",
        "window_minutes": 300,
        "limit_scope": "unified",
        "provider_resets_at": reset_at,
        "reset_hint_seconds": 18000,
        "exhaustion_kind": "usage_limit_reached",
        "account_hash": "acctdeadbeef12",
    }
    candidate = _candidate()
    selection = _selection(candidate, quota_observation=observation)
    decision = await admission.reserve_provider_lane_admission(
        candidate=candidate,
        selection=selection,
        now_epoch=reset_at.timestamp() - 60,
    )
    assert decision.allowed is False
    assert decision.reason == admission.AdmissionDenyReason.CONFIRMED_EXHAUSTED.value
    assert decision.reset_at == reset_at.isoformat()
    assert decision.limit_scope == "unified"
    with pytest.raises(HTTPException) as exc_info:
        admission.raise_provider_lane_admission_rejected(
            decision,
            candidate=candidate,
            alias_model="claude-sonnet-5",
            alias_family="anthropic",
            lane_key=selection["lane_key"],
        )
    exc = exc_info.value
    assert exc.status_code == 429
    detail = exc.detail
    assert detail["error"]["code"] == "aawm_provider_lane_account_exhausted"
    assert detail["error"]["reset_at"] == reset_at.isoformat()
    assert detail["error"]["lane_fingerprint"] == decision.lane_fingerprint
    assert detail["error"]["limit_scope"] == "unified"
    assert detail["admission"]["attempted_provider_call"] is False
    assert exc.headers is not None
    assert exc.headers.get("Retry-After") == "18000"


@pytest.mark.asyncio
async def test_redis_unavailable_degrades_open_without_queue() -> None:
    candidate = _candidate()
    selection = _selection(candidate)

    class _BrokenCache:
        redis_cache = object()

    with patch.object(
        admission.durable,
        "get_aawm_alias_routing_dual_cache",
        return_value=_BrokenCache(),
    ):
        # client/eval missing -> degrade open, no sleep/queue
        decision = await admission.reserve_provider_lane_admission(
            candidate=candidate,
            selection=selection,
            allow_local_fallback=False,
        )
    assert decision.allowed is True
    assert (
        decision.reason
        == admission.AdmissionAdmitReason.REDIS_UNAVAILABLE_DEGRADED.value
    )
    assert decision.lease is None


class _HashAdmissionStore:
    """In-memory stand-in for admission HASH + TTL lease keys."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    def hgetall(self, name: str) -> list[str]:
        mapping = self.data.get(name) or {}
        if not isinstance(mapping, dict):
            return []
        out: list[str] = []
        for key, value in list(mapping.items()):
            out.extend([str(key), str(value)])
        return out

    def hdel(self, name: str, field: str) -> None:
        mapping = self.data.get(name)
        if isinstance(mapping, dict):
            mapping.pop(field, None)
            if not mapping:
                self.data.pop(name, None)

    def hset(self, name: str, field: str, value: Any) -> None:
        mapping = self.data.setdefault(name, {})
        if not isinstance(mapping, dict):
            mapping = {}
            self.data[name] = mapping
        mapping[str(field)] = value

    def lease_alive(self, lease_key: str) -> bool:
        return isinstance(self.data.get(lease_key), str)

    def reclaim_accounting(self, inflight_hash: str) -> dict[str, int]:
        pairs = self.hgetall(inflight_hash)
        accounting = {
            "weighted_units": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "unified_tokens": 0,
            "large_context_count": 0,
        }
        for i in range(0, len(pairs), 2):
            lease_key = pairs[i]
            try:
                payload = json.loads(pairs[i + 1])
                if not isinstance(payload, dict):
                    raise TypeError("accounting payload must be an object")
                weighted_units = int(payload["weighted_units"])
                input_tokens = int(payload["input_tokens"])
                output_tokens = int(payload["output_tokens"])
                unified_tokens = int(payload["unified_tokens"])
                if (
                    weighted_units < 1
                    or input_tokens < 0
                    or output_tokens < 0
                    or unified_tokens < 0
                    or not isinstance(payload.get("reservation_token"), str)
                    or not isinstance(payload.get("large_context"), bool)
                ):
                    raise ValueError("invalid accounting payload")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                self.hdel(inflight_hash, lease_key)
                self.data.pop(lease_key, None)
                self.data.pop(f"ttl:{lease_key}", None)
                self.data.pop(f"token:{lease_key}", None)
                continue
            if not self.lease_alive(lease_key):
                self.hdel(inflight_hash, lease_key)
                self.data.pop(f"ttl:{lease_key}", None)
                self.data.pop(f"token:{lease_key}", None)
                continue
            accounting["weighted_units"] += weighted_units
            accounting["input_tokens"] += input_tokens
            accounting["output_tokens"] += output_tokens
            accounting["unified_tokens"] += unified_tokens
            accounting["large_context_count"] += int(
                payload.get("large_context") is True
            )
        if accounting["weighted_units"] <= 0:
            self.data.pop(inflight_hash, None)
        return accounting


def _make_hash_accounting_fake_redis(store: dict[str, Any] | _HashAdmissionStore):
    """Minimal Redis eval surface matching admission HASH + lease-key Lua."""
    backend = store if isinstance(store, _HashAdmissionStore) else None
    if backend is None:
        # Keep dict compatibility for existing tests by wrapping.
        backend = _HashAdmissionStore()
        backend.data = store  # type: ignore[assignment]
        # actually we need store to be the same object - use shared data ref
    if not isinstance(store, _HashAdmissionStore):
        class _Backend(_HashAdmissionStore):
            def __init__(self, data: dict[str, Any]) -> None:
                self.data = data
        backend = _Backend(store)

    class _FakeRedis:
        async def eval(self, script: str, numkeys: int, *args: Any) -> Any:
            inflight_hash, lease_key = args[0], args[1]
            argv = list(args[numkeys:])
            if "HSET" in script and "max_in_flight" in script:
                max_in_flight = int(argv[0])
                weighted_units = int(argv[1])
                ttl = int(argv[2])
                token = argv[3]
                payload = argv[4]
                max_input_tokens = int(argv[5])
                max_output_tokens = int(argv[6])
                max_unified_tokens = int(argv[7])
                max_large_context_in_flight = int(argv[8])
                try:
                    requested = json.loads(payload)
                    if not isinstance(requested, dict):
                        raise TypeError("accounting payload must be an object")
                    requested_input = int(requested["input_tokens"])
                    requested_output = int(requested["output_tokens"])
                    requested_unified = int(requested["unified_tokens"])
                    requested_weighted = int(requested["weighted_units"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    return [-2, 0]
                if (
                    requested.get("reservation_token") != token
                    or requested_weighted != weighted_units
                    or requested_input < 0
                    or requested_output < 0
                    or requested_unified < 0
                    or not isinstance(requested.get("large_context"), bool)
                ):
                    return [-2, 0]
                if backend.lease_alive(lease_key):
                    return [-3, 0]
                current = backend.reclaim_accounting(inflight_hash)
                denial_checks = (
                    (
                        0,
                        current["weighted_units"],
                        weighted_units,
                        max_in_flight,
                    ),
                    (
                        -4,
                        current["input_tokens"],
                        requested_input,
                        max_input_tokens,
                    ),
                    (
                        -5,
                        current["output_tokens"],
                        requested_output,
                        max_output_tokens,
                    ),
                    (
                        -6,
                        current["unified_tokens"],
                        requested_unified,
                        max_unified_tokens,
                    ),
                )
                for code, existing, added, maximum in denial_checks:
                    if existing + added > maximum:
                        return [code, current["weighted_units"]]
                if (
                    requested.get("large_context") is True
                    and current["large_context_count"] + 1
                    > max_large_context_in_flight
                ):
                    return [-7, current["weighted_units"]]
                backend.hset(inflight_hash, lease_key, payload)
                backend.data[lease_key] = payload
                backend.data[f"ttl:{lease_key}"] = ttl
                backend.data[f"token:{lease_key}"] = token
                return [1, current["weighted_units"] + weighted_units]
            if "reservation_token" in script and "HDEL" in script:
                token = argv[0]
                raw = backend.data.get(lease_key)
                if isinstance(raw, str):
                    try:
                        current = json.loads(raw)
                    except json.JSONDecodeError:
                        backend.data.pop(lease_key, None)
                    else:
                        if not isinstance(current, dict):
                            backend.data.pop(lease_key, None)
                        elif current.get("reservation_token") == token:
                            backend.data.pop(lease_key, None)
                            backend.hdel(inflight_hash, lease_key)
                    if lease_key not in backend.data:
                        backend.data.pop(f"ttl:{lease_key}", None)
                        backend.data.pop(f"token:{lease_key}", None)
                remaining = backend.reclaim_accounting(inflight_hash)[
                    "weighted_units"
                ]
                return [1, remaining]
            raise AssertionError(f"unexpected lua script: {script[:80]}")

    return _FakeRedis()


@pytest.mark.asyncio
async def test_lua_reserve_release_via_fake_redis_eval() -> None:
    """Atomic reservation behavior at the component Redis/Lua seam."""

    store: dict[str, Any] = {}

    class _RedisCache:
        def init_async_client(self):
            return _make_hash_accounting_fake_redis(store)

        def check_and_fix_namespace(self, key: str) -> str:
            return f"ns:{key}"

    class _Dual:
        redis_cache = _RedisCache()

    candidate = _candidate()
    selection = _selection(candidate)
    with patch.object(
        admission.durable,
        "get_aawm_alias_routing_dual_cache",
        return_value=_Dual(),
    ):
        first = await admission.reserve_provider_lane_admission(
            candidate=candidate,
            selection=selection,
            max_in_flight=1,
            lease_ttl_seconds=30,
            allow_local_fallback=False,
        )
        assert first.allowed is True
        assert first.reason == admission.AdmissionAdmitReason.RESERVED.value
        assert first.lease is not None
        assert first.lease.durable is True
        second = await admission.reserve_provider_lane_admission(
            candidate=candidate,
            selection=selection,
            max_in_flight=1,
            lease_ttl_seconds=30,
            allow_local_fallback=False,
        )
        assert second.allowed is False
        released = await admission.release_provider_lane_admission(first.lease)
        assert released is True
        third = await admission.reserve_provider_lane_admission(
            candidate=candidate,
            selection=selection,
            max_in_flight=1,
            lease_ttl_seconds=30,
            allow_local_fallback=False,
        )
        assert third.allowed is True
        await admission.release_provider_lane_admission(third.lease)


@pytest.mark.asyncio
async def test_admission_separated_from_cooldown_and_session_ownership() -> None:
    """Admission must not read/write cooldown maps or session-owner records."""
    candidate = _candidate()
    selection = _selection(candidate)
    family = alias_routing_state.family("anthropic")
    cooldown_key = selection["cooldown_key"]
    # Use the real cooldown memory API without going through admission.
    family.set_cooldown_memory(cooldown_key, 999.0)
    before_remaining = family.get_memory_cooldown_remaining(cooldown_key)

    with patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.session_affinity.guard_session_owner_before_egress",
        new_callable=AsyncMock,
    ) as guard_mock, patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.session_affinity.get_session_owner_record",
        new_callable=AsyncMock,
    ) as owner_mock:
        decision = await admission.reserve_provider_lane_admission(
            candidate=candidate,
            selection=selection,
            max_in_flight=2,
        )
        assert decision.allowed is True
        assert decision.lease is not None
        await admission.release_provider_lane_admission(decision.lease)
        guard_mock.assert_not_called()
        owner_mock.assert_not_called()

    after_remaining = family.get_memory_cooldown_remaining(cooldown_key)
    # Admission must not clear/replace alias cooldown state.
    assert after_remaining > 0
    assert cooldown_key in family.cooldown_until_monotonic_by_key
    # Remaining only decays with wall clock; allow generous skew for slow hosts.
    assert after_remaining <= before_remaining + 0.001
    family.clear_cooldown_state(cooldown_keys=[cooldown_key])


def _make_services(
    *,
    selection: dict[str, Any],
    perform_response: Optional[Response] = None,
    perform_exc: Optional[Exception] = None,
    perform_mock: Optional[AsyncMock] = None,
) -> tuple[AliasRouteServices, AsyncMock]:
    if perform_mock is None:
        perform_mock = AsyncMock()
        if perform_exc is not None:
            perform_mock.side_effect = perform_exc
        else:
            perform_mock.return_value = perform_response or Response(
                content=b"ok", status_code=200
            )

    async def _select(*, request: Any, request_body: Any) -> dict[str, Any]:
        return dict(selection)

    async def _noop_persist(*, keys, seconds):  # noqa: ANN001
        return None

    def _publish(*, keys, seconds):  # noqa: ANN001
        return None

    async def _set_affinity(session_key, candidate):  # noqa: ANN001
        return None

    def _add_meta(body, *, request, selection, attempts):  # noqa: ANN001
        out = dict(body)
        out["litellm_metadata"] = {
            "attempts": attempts,
            "selection_reason": selection.get("selection_reason"),
        }
        return out

    def _resolve(**kwargs):  # noqa: ANN001
        return _plan()

    def _raise_redispatch(**kwargs):  # noqa: ANN001
        raise HTTPException(status_code=429, detail={"error": {"code": "redispatch"}})

    services = AliasRouteServices(
        select_candidate_fn=_select,
        perform_candidate_request_fn=perform_mock,
        resolve_cooldown_publication_fn=_resolve,
        publish_cooldown_memory_fn=_publish,
        persist_cooldown_fn=_noop_persist,
        set_session_affinity_fn=_set_affinity,
        add_alias_metadata_fn=_add_meta,
        raise_redispatch_fn=_raise_redispatch,
    )
    return services, perform_mock


@pytest.mark.asyncio
async def test_candidate_loop_runs_admission_before_provider_io() -> None:
    """Pre-I/O placement: denial happens before perform_candidate_request_fn."""
    reset_at = datetime(2026, 8, 11, 20, 0, tzinfo=timezone.utc)
    candidate = _candidate(
        codex_oauth_account_hash=None,
        codex_oauth_account_label=None,
        codex_oauth_lane_key=None,
        provider_account_hash="acctdeadbeef12",
    )
    selection = _selection(candidate)
    services, perform_mock = _make_services(selection=selection)
    request = _minimal_request()

    denied = admission.AdmissionDecision(
        allowed=False,
        reason=admission.AdmissionDenyReason.CONFIRMED_EXHAUSTED.value,
        lane_fingerprint="fp-test",
        provider="anthropic",
        account_hash="acctdeadbeef12",
        limit_scope="unified",
        reset_at=reset_at.isoformat(),
        reset_hint_seconds=60,
        exhaustion_kind="usage_limit_reached",
        detail_code="aawm_provider_lane_account_exhausted",
    )

    class _AdmissionProxy:
        async def admit_selected_candidate(self, **kwargs):
            attempt_record = kwargs.get("attempt_record")
            if isinstance(attempt_record, dict):
                attempt_record["status"] = "admission_denied"
                attempt_record["attempted_provider_call"] = False
            return denied

        def admission_deny_error_class(self, decision):
            return admission.admission_deny_error_class(decision)

        def raise_provider_lane_admission_rejected(self, *args, **kwargs):
            return admission.raise_provider_lane_admission_rejected(*args, **kwargs)

        async def release_provider_lane_admission(self, lease):
            return False

    with patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.candidate_loop._admission_mod",
        return_value=_AdmissionProxy(),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await handle_alias_route(
                services,
                alias_family="anthropic",
                alias_model="claude-sonnet-5",
                request=request,
                prepared_request_body={"model": "claude-sonnet-5", "messages": []},
                max_candidate_attempts=1,
                get_active_cooldown_state_fn=AsyncMock(return_value=(0.0, "none")),
                attempts_metadata_key="aawm_alias_attempts",
                skipped_candidates_metadata_key="aawm_alias_skipped",
                no_candidate_detail="none",
                log_label="Anthropic",
            )
    assert exc_info.value.status_code == 429
    detail = exc_info.value.detail
    assert isinstance(detail, dict), detail
    assert detail["error"]["code"] == "aawm_provider_lane_account_exhausted"
    assert detail["error"]["reset_at"] == reset_at.isoformat()
    perform_mock.assert_not_called()



@pytest.mark.asyncio
async def test_candidate_loop_releases_admission_lease_after_success() -> None:
    candidate = _candidate(
        codex_oauth_account_hash=None,
        codex_oauth_account_label=None,
        codex_oauth_lane_key=None,
        provider_account_hash="acctsuccess01",
    )
    selection = _selection(
        candidate,
        cooldown_key="admission-success-unique-cooldown",
        lane_key="admission-success-lane",
    )
    services, perform_mock = _make_services(selection=selection)
    request = _minimal_request()

    call_order: list[str] = []

    real_admit = admission.admit_selected_candidate
    real_release = admission.release_provider_lane_admission

    async def _admit(**kwargs):
        call_order.append("admit")
        return await real_admit(**kwargs)

    async def _release(lease):
        call_order.append("release")
        return await real_release(lease)

    async def _perform(**kwargs):
        call_order.append("provider_io")
        return Response(content=b"ok", status_code=200)

    perform_mock.side_effect = _perform

    sa = MagicMock()
    sa.resolve_canonical_session_identity.return_value = "sess"
    sa.build_session_owner_attributes.return_value = {"provider": "anthropic"}
    guard = SimpleNamespace(
        decision=SimpleNamespace(value="owned"),
        reservation_token=None,
        held_reservation=False,
        provenance=None,
    )
    sa.ensure_session_owner_guard_for_request = AsyncMock(return_value=guard)
    sa.get_request_session_owner_lease.return_value = None
    sa.finalize_session_owner_lease_on_success = AsyncMock(return_value=None)
    sa.finalize_session_owner_lease_on_failure = AsyncMock(return_value=None)

    with patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.candidate_loop._session_affinity_mod",
        return_value=sa,
    ), patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.candidate_loop._admission_mod",
    ) as adm_mod:
        # Use real admission module object but spy admit/release through wrapper.
        import litellm.proxy.pass_through_endpoints.aawm_alias_routing.admission as adm

        class _Proxy:
            async def admit_selected_candidate(self, **kwargs):
                return await _admit(**kwargs)

            async def release_provider_lane_admission(self, lease):
                return await _release(lease)

            def admission_deny_error_class(self, decision):
                return adm.admission_deny_error_class(decision)

            def raise_provider_lane_admission_rejected(self, *a, **k):
                return adm.raise_provider_lane_admission_rejected(*a, **k)

        adm_mod.return_value = _Proxy()
        response = await handle_alias_route(
            services,
            alias_family="anthropic",
            alias_model="claude-sonnet-5",
            request=request,
            prepared_request_body={"model": "claude-sonnet-5", "messages": []},
            max_candidate_attempts=1,
            get_active_cooldown_state_fn=AsyncMock(return_value=(0.0, "none")),
            attempts_metadata_key="aawm_alias_attempts",
            skipped_candidates_metadata_key="aawm_alias_skipped",
            no_candidate_detail="none",
            log_label="Anthropic",
        )
    assert response.status_code == 200
    assert call_order == ["admit", "provider_io", "release"]
    # Lease must not remain held after success.
    assert admission._local_inflight_by_lane == {}


def test_handle_alias_route_source_orders_admission_before_attempt_start() -> None:
    source = inspect.getsource(handle_alias_route)
    admit_at = source.index("await admission.admit_selected_candidate")
    # First live call site after the late local alias assignment.
    assign_at = source.index("_record_auto_agent_alias_attempt_started = _lpe._record_auto_agent_alias_attempt_started")
    started_at = source.index(
        "candidate_body = _record_auto_agent_alias_attempt_started(",
        assign_at,
    )
    perform_at = source.index("response = await perform_candidate_request_fn(")
    assert admit_at < started_at < perform_at


@pytest.mark.asyncio
async def test_expired_lease_reclaimed_under_continuing_reservations() -> None:
    """Expired/missing lease keys must not keep hash units forever under new traffic."""
    store: dict[str, Any] = {}
    candidate = _candidate(provider_account_hash="acct-reclaim-1")
    # strip oauth fields to keep fingerprint on provider_account_hash
    candidate.pop("codex_oauth_account_hash", None)
    candidate.pop("codex_oauth_account_label", None)
    candidate.pop("codex_oauth_lane_key", None)
    selection = _selection(candidate, lane_key="reclaim-lane")

    class _RedisCache:
        def init_async_client(self):
            return _make_hash_accounting_fake_redis(store)

        def check_and_fix_namespace(self, key: str) -> str:
            return f"ns:{key}"

    class _Dual:
        redis_cache = _RedisCache()

    with patch.object(
        admission.durable,
        "get_aawm_alias_routing_dual_cache",
        return_value=_Dual(),
    ):
        first = await admission.reserve_provider_lane_admission(
            candidate=candidate,
            selection=selection,
            max_in_flight=1,
            lease_ttl_seconds=30,
            allow_local_fallback=False,
        )
        assert first.allowed is True and first.lease is not None
        # Simulate TTL expiry of the lease key while hash residue remains
        # (and would otherwise be kept alive by newer EXPIRE on the hash).
        lease_key = f"ns:{first.lease.lease_cache_key}"
        inflight_hash = f"ns:{first.lease.counter_cache_key}"
        assert lease_key in store
        store.pop(lease_key, None)
        assert isinstance(store.get(inflight_hash), dict)
        assert store[inflight_hash]

        # Continuing reservation must reclaim the vanished lease and admit.
        second = await admission.reserve_provider_lane_admission(
            candidate=candidate,
            selection=selection,
            max_in_flight=1,
            lease_ttl_seconds=30,
            allow_local_fallback=False,
        )
        assert second.allowed is True, second
        assert second.lease is not None
        await admission.release_provider_lane_admission(second.lease)


@pytest.mark.asyncio
async def test_local_expired_lease_reclaimed_before_capacity_check() -> None:
    candidate = _candidate()
    selection = _selection(candidate)
    now = 1_700_000_000.0
    first = await admission.reserve_provider_lane_admission(
        candidate=candidate,
        selection=selection,
        max_in_flight=1,
        lease_ttl_seconds=10,
        now_epoch=now,
    )
    assert first.allowed is True
    # Advance past expiry; next reserve must reclaim rather than deny forever.
    second = await admission.reserve_provider_lane_admission(
        candidate=candidate,
        selection=selection,
        max_in_flight=1,
        lease_ttl_seconds=10,
        now_epoch=now + 11,
    )
    assert second.allowed is True
    await admission.release_provider_lane_admission(second.lease)


@pytest.mark.asyncio
async def test_pre_attempt_exception_releases_admission_lease() -> None:
    """Exceptions from attempt recording before probe/provider must release."""
    selection = _selection(
        _candidate(
            codex_oauth_account_hash=None,
            codex_oauth_account_label=None,
            codex_oauth_lane_key=None,
            provider_account_hash="acct-pre-attempt",
        ),
        cooldown_key="admission-pre-attempt-cd",
        lane_key="admission-pre-attempt-lane",
    )
    services, perform_mock = _make_services(selection=selection)
    request = _minimal_request()

    releases: list[Any] = []
    real_admit = admission.admit_selected_candidate

    class _Proxy:
        async def admit_selected_candidate(self, **kwargs):
            return await real_admit(**kwargs)

        async def release_provider_lane_admission(self, lease):
            releases.append(lease)
            return await admission.release_provider_lane_admission(lease)

        def admission_deny_error_class(self, decision):
            return admission.admission_deny_error_class(decision)

        def raise_provider_lane_admission_rejected(self, *a, **k):
            return admission.raise_provider_lane_admission_rejected(*a, **k)

    with patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.candidate_loop._admission_mod",
        return_value=_Proxy(),
    ), patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._record_auto_agent_alias_attempt_started",
        side_effect=RuntimeError("attempt-record-boom"),
    ):
        with pytest.raises(RuntimeError, match="attempt-record-boom"):
            await handle_alias_route(
                services,
                alias_family="anthropic",
                alias_model="claude-sonnet-5",
                request=request,
                prepared_request_body={"model": "claude-sonnet-5", "messages": []},
                max_candidate_attempts=1,
                get_active_cooldown_state_fn=AsyncMock(return_value=(0.0, "none")),
                attempts_metadata_key="aawm_alias_attempts",
                skipped_candidates_metadata_key="aawm_alias_skipped",
                no_candidate_detail="none",
                log_label="Anthropic",
            )
    assert len(releases) == 1
    assert releases[0] is not None
    perform_mock.assert_not_called()
    assert admission._local_inflight_by_lane == {}


@pytest.mark.asyncio
async def test_cancellation_during_cooldown_precheck_releases_admission_lease() -> None:
    """Cancellation between admission and provider I/O must not leak the lease."""
    selection = _selection(
        _candidate(
            codex_oauth_account_hash=None,
            codex_oauth_account_label=None,
            codex_oauth_lane_key=None,
            provider_account_hash="acct-cancel",
        ),
        cooldown_key="admission-cancel-cd",
        lane_key="admission-cancel-lane",
    )
    services, perform_mock = _make_services(selection=selection)
    request = _minimal_request()
    releases: list[Any] = []
    real_admit = admission.admit_selected_candidate

    class _Proxy:
        async def admit_selected_candidate(self, **kwargs):
            return await real_admit(**kwargs)

        async def release_provider_lane_admission(self, lease):
            releases.append(lease)
            return await admission.release_provider_lane_admission(lease)

        def admission_deny_error_class(self, decision):
            return admission.admission_deny_error_class(decision)

        def raise_provider_lane_admission_rejected(self, *a, **k):
            return admission.raise_provider_lane_admission_rejected(*a, **k)

    async def _cancel_cooldown(_key: str):
        raise asyncio.CancelledError()

    # Need asyncio import for CancelledError - add if missing
    with patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.candidate_loop._admission_mod",
        return_value=_Proxy(),
    ):
        with pytest.raises(asyncio.CancelledError):
            await handle_alias_route(
                services,
                alias_family="anthropic",
                alias_model="claude-sonnet-5",
                request=request,
                prepared_request_body={"model": "claude-sonnet-5", "messages": []},
                max_candidate_attempts=1,
                get_active_cooldown_state_fn=_cancel_cooldown,
                attempts_metadata_key="aawm_alias_attempts",
                skipped_candidates_metadata_key="aawm_alias_skipped",
                no_candidate_detail="none",
                log_label="Anthropic",
            )
    assert len(releases) == 1
    perform_mock.assert_not_called()
    assert admission._local_inflight_by_lane == {}


def test_token_estimates_and_trusted_interactive_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AAWM_PROVIDER_LANE_ADMISSION_TOKEN_WEIGHT_QUANTUM", "100")
    monkeypatch.setenv("AAWM_PROVIDER_LANE_ADMISSION_LARGE_CONTEXT_TOKENS", "200")
    candidate = _candidate(
        provider="openai",
        model="gpt-5",
        route_family="openai_responses",
    )
    selection = _selection(
        candidate,
        token_estimates={"input_tokens": 150, "output_tokens": 75},
        request_class="interactive",
        in_flight_session=False,
    )

    assert admission.estimate_provider_lane_tokens(
        candidate=candidate,
        selection=selection,
    ) == (150, 75, 225)
    assert admission.weighted_token_units(225) == 3
    assert admission.resolve_large_context(
        candidate=candidate,
        selection=selection,
        unified_tokens=225,
    ) is True
    assert admission.resolve_reserved_interactive(
        candidate=candidate,
        selection=selection,
    ) is False

    trusted_selection = _selection(
        candidate,
        token_estimates={"input_tokens": 1, "output_tokens": 1},
        request_class="standard",
        in_flight_session=True,
    )
    assert admission.resolve_reserved_interactive(
        candidate=candidate,
        selection=trusted_selection,
    ) is True


@pytest.mark.asyncio
async def test_local_token_accounting_caps_and_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AAWM_PROVIDER_LANE_ADMISSION_TOKEN_WEIGHT_QUANTUM", "100")
    monkeypatch.setenv("AAWM_PROVIDER_LANE_ADMISSION_MAX_INPUT_TOKENS", "200")
    monkeypatch.setenv("AAWM_PROVIDER_LANE_ADMISSION_MAX_OUTPUT_TOKENS", "100")
    monkeypatch.setenv("AAWM_PROVIDER_LANE_ADMISSION_MAX_UNIFIED_TOKENS", "250")
    candidate = _candidate(
        provider="openai",
        model="gpt-5",
        route_family="openai_responses",
    )
    selection = _selection(
        candidate,
        token_estimates={"input_tokens": 150, "output_tokens": 50},
    )

    first = await admission.reserve_provider_lane_admission(
        candidate=candidate,
        selection=selection,
        max_in_flight=8,
    )
    assert first.allowed is True
    assert first.input_tokens == 150
    assert first.output_tokens == 50
    assert first.unified_tokens == 200
    assert first.weighted_units == 2
    assert first.capacity_class == "standard"
    assert first.lease is not None
    assert first.lease.input_tokens == 150
    assert first.lease.weighted_units == 2

    second = await admission.reserve_provider_lane_admission(
        candidate=candidate,
        selection=selection,
        max_in_flight=8,
    )
    assert second.allowed is False
    assert second.reason == admission.AdmissionDenyReason.CAPACITY_UNAVAILABLE.value
    assert second.limit_scope == "input"
    detail = admission.build_admission_rejection_detail(
        second,
        candidate=candidate,
        alias_model="gpt-5",
        alias_family="openai",
        lane_key=selection["lane_key"],
    )
    assert detail["error"]["provider"] == "openai"
    assert detail["error"]["limit_scope"] == "input"
    assert detail["admission"]["attempted_provider_call"] is False

    assert await admission.release_provider_lane_admission(first.lease) is True
    third = await admission.reserve_provider_lane_admission(
        candidate=candidate,
        selection=selection,
        max_in_flight=8,
    )
    assert third.allowed is True
    await admission.release_provider_lane_admission(third.lease)


@pytest.mark.asyncio
async def test_large_context_has_separate_bounded_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AAWM_PROVIDER_LANE_ADMISSION_LARGE_CONTEXT_TOKENS", "100")
    monkeypatch.setenv("AAWM_PROVIDER_LANE_ADMISSION_LARGE_CONTEXT_MAX_IN_FLIGHT", "1")
    candidate = _candidate(
        provider="openai",
        model="gpt-5",
        route_family="openai_responses",
    )
    selection = _selection(
        candidate,
        token_estimates={"input_tokens": 150, "output_tokens": 25},
    )
    first = await admission.reserve_provider_lane_admission(
        candidate=candidate,
        selection=selection,
        max_in_flight=4,
    )
    assert first.allowed is True
    assert first.large_context is True
    assert first.max_large_context_in_flight == 1

    second = await admission.reserve_provider_lane_admission(
        candidate=candidate,
        selection=selection,
        max_in_flight=4,
    )
    assert second.allowed is False
    assert second.large_context is True
    assert second.limit_scope == "concurrency"
    assert second.detail_code == "aawm_provider_lane_large_context_capacity_unavailable"
    await admission.release_provider_lane_admission(first.lease)


@pytest.mark.asyncio
async def test_reserved_interactive_capacity_is_not_caller_selectable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "AAWM_PROVIDER_LANE_ADMISSION_RESERVED_INTERACTIVE_CAPACITY", "1"
    )
    candidate = _candidate(
        provider="openai",
        model="gpt-5",
        route_family="openai_responses",
    )
    standard_selection = _selection(
        candidate,
        token_estimates={"input_tokens": 1, "output_tokens": 1},
        request_class="standard",
        in_flight_session=False,
    )
    first = await admission.reserve_provider_lane_admission(
        candidate=candidate,
        selection=standard_selection,
        max_in_flight=2,
    )
    assert first.allowed is True

    second = await admission.reserve_provider_lane_admission(
        candidate=candidate,
        selection=standard_selection,
        max_in_flight=2,
    )
    assert second.allowed is False

    trusted_selection = _selection(
        candidate,
        token_estimates={"input_tokens": 1, "output_tokens": 1},
        request_class="standard",
        in_flight_session=True,
    )
    trusted = await admission.reserve_provider_lane_admission(
        candidate=candidate,
        selection=trusted_selection,
        max_in_flight=2,
    )
    assert trusted.allowed is True
    assert trusted.reserved_interactive is True
    await admission.release_provider_lane_admission(first.lease)
    await admission.release_provider_lane_admission(trusted.lease)


@pytest.mark.asyncio
async def test_redis_accounting_payload_release_and_expired_reclaim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store: dict[str, Any] = {}
    monkeypatch.setenv("AAWM_PROVIDER_LANE_ADMISSION_TOKEN_WEIGHT_QUANTUM", "100")
    monkeypatch.setenv("AAWM_PROVIDER_LANE_ADMISSION_MAX_INPUT_TOKENS", "60")
    monkeypatch.setenv("AAWM_PROVIDER_LANE_ADMISSION_MAX_OUTPUT_TOKENS", "100")
    monkeypatch.setenv("AAWM_PROVIDER_LANE_ADMISSION_MAX_UNIFIED_TOKENS", "200")
    monkeypatch.setenv("AAWM_PROVIDER_LANE_ADMISSION_LARGE_CONTEXT_TOKENS", "1000")

    class _RedisCache:
        def init_async_client(self):
            return _make_hash_accounting_fake_redis(store)

        def check_and_fix_namespace(self, key: str) -> str:
            return f"ns:{key}"

    class _Dual:
        redis_cache = _RedisCache()

    candidate = _candidate(
        provider="openai",
        model="gpt-5",
        route_family="openai_responses",
    )
    selection = _selection(
        candidate,
        token_estimates={"input_tokens": 40, "output_tokens": 20},
    )
    with patch.object(
        admission.durable,
        "get_aawm_alias_routing_dual_cache",
        return_value=_Dual(),
    ):
        first = await admission.reserve_provider_lane_admission(
            candidate=candidate,
            selection=selection,
            max_in_flight=2,
            allow_local_fallback=False,
        )
        assert first.allowed is True
        assert first.lease is not None
        payload = json.loads(store[f"ns:{first.lease.lease_cache_key}"])
        assert payload["input_tokens"] == 40
        assert payload["output_tokens"] == 20
        assert payload["unified_tokens"] == 60
        assert payload["weighted_units"] >= 1

        lease_key = f"ns:{first.lease.lease_cache_key}"
        inflight_hash = f"ns:{first.lease.counter_cache_key}"
        assert store[inflight_hash][lease_key] == store[lease_key]
        denied_input = await admission.reserve_provider_lane_admission(
            candidate=candidate,
            selection=selection,
            max_in_flight=2,
            allow_local_fallback=False,
        )
        assert denied_input.allowed is False
        assert denied_input.limit_scope == "input"
        assert denied_input.detail_code == "aawm_provider_lane_capacity_unavailable"

        store.pop(lease_key, None)
        reclaimed = await admission.reserve_provider_lane_admission(
            candidate=candidate,
            selection=selection,
            max_in_flight=2,
            allow_local_fallback=False,
        )
        assert reclaimed.allowed is True
        assert reclaimed.lease is not None
        assert await admission.release_provider_lane_admission(reclaimed.lease) is True

        monkeypatch.setenv("AAWM_PROVIDER_LANE_ADMISSION_MAX_INPUT_TOKENS", "1000")
        monkeypatch.setenv("AAWM_PROVIDER_LANE_ADMISSION_MAX_OUTPUT_TOKENS", "1000")
        monkeypatch.setenv("AAWM_PROVIDER_LANE_ADMISSION_MAX_UNIFIED_TOKENS", "1000")
        monkeypatch.setenv("AAWM_PROVIDER_LANE_ADMISSION_LARGE_CONTEXT_TOKENS", "50")
        monkeypatch.setenv(
            "AAWM_PROVIDER_LANE_ADMISSION_LARGE_CONTEXT_MAX_IN_FLIGHT", "1"
        )
        large_first = await admission.reserve_provider_lane_admission(
            candidate=candidate,
            selection=selection,
            max_in_flight=2,
            allow_local_fallback=False,
        )
        assert large_first.allowed is True
        assert large_first.lease is not None
        denied_large = await admission.reserve_provider_lane_admission(
            candidate=candidate,
            selection=selection,
            max_in_flight=2,
            allow_local_fallback=False,
        )
        assert denied_large.allowed is False
        assert denied_large.limit_scope == "concurrency"
        assert (
            denied_large.detail_code
            == "aawm_provider_lane_large_context_capacity_unavailable"
        )
        assert await admission.release_provider_lane_admission(large_first.lease) is True
        large_after_release = await admission.reserve_provider_lane_admission(
            candidate=candidate,
            selection=selection,
            max_in_flight=2,
            allow_local_fallback=False,
        )
        assert large_after_release.allowed is True
        assert large_after_release.lease is not None
        assert (
            await admission.release_provider_lane_admission(large_after_release.lease)
            is True
        )
        assert not store.get(inflight_hash)


@pytest.mark.asyncio
async def test_denial_warning_emits_first_and_bounded_summary(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv(
        "AAWM_PROVIDER_LANE_ADMISSION_WARNING_SUMMARY_INTERVAL_SECONDS", "10"
    )
    candidate = _candidate(
        provider="openai",
        model="gpt-5",
        route_family="openai_responses",
    )
    selection = _selection(
        candidate,
        token_estimates={"input_tokens": 1, "output_tokens": 1},
        client_id="client-1",
        alias_model="gpt-5",
        alias_family="openai",
    )
    caplog.set_level("WARNING", logger="LiteLLM Proxy")
    first = await admission.reserve_provider_lane_admission(
        candidate=candidate,
        selection=selection,
        max_in_flight=1,
        now_epoch=100,
    )
    assert first.allowed is True
    for now in (101, 105, 109):
        denied = await admission.reserve_provider_lane_admission(
            candidate=candidate,
            selection=selection,
            max_in_flight=1,
            now_epoch=now,
        )
        assert denied.allowed is False
    assert sum("admission warning:" in record.message for record in caplog.records) == 1

    denied = await admission.reserve_provider_lane_admission(
        candidate=candidate,
        selection=selection,
        max_in_flight=1,
        now_epoch=111,
    )
    assert denied.allowed is False
    assert sum(
        "admission warning summary:" in record.message for record in caplog.records
    ) == 1
    assert any("openai" in record.message for record in caplog.records)
    await admission.release_provider_lane_admission(first.lease)
