"""D1-612 boundary tests: tokenized pre-egress session ownership."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import redis.exceptions
from fastapi import HTTPException

from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import durable as durable_mod
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import session_affinity as sa


class _FakeRedisClient:
    def __init__(self, parent: "_FakeRedisCache") -> None:
        self._parent = parent

    async def get(self, name: str) -> Any:
        self._parent._purge_expired(name)
        return self._parent._data.get(name)

    async def set(self, name: str, value: Any, nx: bool = False, ex: Any = None) -> bool:
        self._parent._purge_expired(name)
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
            ttl = float(ex)
            self._parent._ttl[name] = ttl
            self._parent._expires_at[name] = self._parent._clock + ttl
        else:
            self._parent._ttl.pop(name, None)
            self._parent._expires_at.pop(name, None)
        return True

    async def delete(self, *names: str) -> int:
        deleted = 0
        for name in names:
            if name in self._parent._data:
                self._parent._data.pop(name, None)
                self._parent._ttl.pop(name, None)
                self._parent._expires_at.pop(name, None)
                deleted += 1
        return deleted

    async def persist(self, name: str) -> bool:
        self._parent._ttl.pop(name, None)
        self._parent._expires_at.pop(name, None)
        return True

    async def eval(self, script: str, numkeys: int, *args: Any) -> Any:
        # Minimal Lua semantics used by session_affinity promote/release/renew.
        key = args[0]
        self._parent._purge_expired(key)
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
            self._parent._expires_at.pop(key, None)
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
                self._parent._expires_at.pop(key, None)
                return 1
            return 0
        if "last_renewed_at_epoch" in script or ("EX" in script and "reserved" in script):
            self._parent.renewal_calls += 1
            token = args[1]
            payload_json = args[2]
            ttl = float(args[3])
            if raw is None:
                return 0
            if self._parent.drop_reservation_on_renewal:
                self._parent._data.pop(key, None)
                self._parent._ttl.pop(key, None)
                self._parent._expires_at.pop(key, None)
                return 0
            if self._parent.fail_renewal:
                return 0
            current = json.loads(raw.decode("utf-8"))
            if current.get("state") != "reserved" or current.get("reservation_token") != token:
                return 0
            payload = json.loads(payload_json)
            self._parent._data[key] = json.dumps(payload).encode("utf-8")
            self._parent._ttl[key] = ttl
            self._parent._expires_at[key] = self._parent._clock + ttl
            return 1
        raise AssertionError(f"unexpected eval script: {script[:80]}")


class _FakeRedisCache:
    def __init__(self, namespace: str = "litellm") -> None:
        self.namespace = namespace
        self._data: dict[str, bytes] = {}
        self._ttl: dict[str, float] = {}
        self._expires_at: dict[str, float] = {}
        self._clock = 0.0
        self.set_calls: list[dict[str, Any]] = []
        self.renewal_calls = 0
        self.fail_renewal = False
        self.drop_reservation_on_renewal = False
        self.fail_set_error: Optional[BaseException] = None
        self._client = _FakeRedisClient(self)

    def _purge_expired(self, name: str) -> None:
        expires_at = self._expires_at.get(name)
        if expires_at is None or expires_at > self._clock:
            return
        self._data.pop(name, None)
        self._ttl.pop(name, None)
        self._expires_at.pop(name, None)

    def advance_time(self, seconds: float) -> None:
        self._clock += float(seconds)

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


def _full_adapter_reasoning_item(
    *,
    summary: Any = None,
) -> dict[str, Any]:
    return {
        "type": "reasoning",
        "id": "rs_adapter_item",
        "summary": summary,
        "encrypted_content": None,
        "content": None,
        "internal_chat_message_metadata_passthrough": {
            "turn_id": "turn-adapter-item",
        },
    }


async def _seed_nonheld_compatible_owner_lease(
    *,
    request: Any,
    session_identity: str,
    attributes: dict[str, Any],
) -> sa.SessionOwnerLease:
    reserved = await sa.guard_session_owner_before_egress(
        session_identity=session_identity,
        requested_attributes=attributes,
    )
    assert reserved.decision is sa.SessionOwnerGuardDecision.UNOWNED_RESERVED
    promoted = await sa.promote_session_owner_reservation(
        session_identity=session_identity,
        reservation_token=reserved.reservation_token,
        attributes=attributes,
    )
    assert promoted.outcome is sa.SessionOwnerMutationOutcome.PROMOTED
    continuation = await sa.ensure_session_owner_guard_for_request(
        request=request,
        session_identity=session_identity,
        requested_attributes=attributes,
        require_exact_attributes=True,
        raise_on_redispatch=False,
    )
    assert continuation.decision is sa.SessionOwnerGuardDecision.COMPATIBLE_OWNER
    lease = sa.get_request_session_owner_lease(request)
    assert lease is not None
    assert lease.held_reservation is False
    return lease


def _redis_snapshot(redis: _FakeRedisCache) -> tuple[Any, Any, Any]:
    return dict(redis._data), dict(redis._ttl), dict(redis._expires_at)


@pytest.mark.parametrize(
    "summary",
    [
        pytest.param(None, id="null-summary"),
        pytest.param(
            [{"type": "summary_text", "text": "local adapter summary"}],
            id="nonempty-summary",
        ),
    ],
)
def test_replay_safety_accepts_full_adapter_reasoning_item(
    summary: Any,
) -> None:
    body = {"input": [_full_adapter_reasoning_item(summary=summary)]}

    result = sa.classify_session_owner_replay_safety_body(body)

    assert result.safe is True
    assert result.field_path is None
    assert result.classification is None
    assert sa.is_replay_safe_session_owner_redispatch_body(body) is True


def test_replay_safety_accepts_encrypted_self_contained_reasoning_item() -> None:
    body = {
        "input": [
            {
                "type": "reasoning",
                "id": "rs_encrypted_item",
                "encrypted_content": "portable-encrypted-state",
            }
        ]
    }

    result = sa.classify_session_owner_replay_safety_body(body)

    assert result.safe is True
    assert sa.is_replay_safe_session_owner_redispatch_body(body) is True


@pytest.mark.parametrize(
    ("request_body", "field_path", "classification"),
    [
        pytest.param(
            {"previous_response_id": "resp_provider_state"},
            "$.previous_response_id",
            "previous_response_id",
            id="top-level-previous-response-id",
        ),
        pytest.param(
            {"metadata": [{"previous_response_id": "rs_provider_state"}]},
            "$.*[0].previous_response_id",
            "previous_response_id",
            id="nested-previous-response-id",
        ),
        pytest.param(
            {"input": [{"type": "reasoning", "id": "rs_provider_item"}]},
            "$.*[0].id",
            "id_only_reasoning_reference",
            id="id-only-reasoning",
        ),
        pytest.param(
            {
                "attacker-secret-ancestor": {
                    "nested-secret-ancestor": [
                        {"provider_item_id": "rs_secret_reference"}
                    ]
                }
            },
            "$.*.*[0].provider_item_id",
            "explicit_item_reference",
            id="attacker-controlled-ancestor-keys",
        ),
        *[
            pytest.param(
                {"input": [{"type": "message", field: "rs_provider_reference"}]},
                f"$.*[0].{field}",
                "explicit_item_reference",
                id=field,
            )
            for field in (
                "item_id",
                "item_reference",
                "provider_item_id",
                "response_item_id",
            )
        ],
    ],
)
def test_replay_safety_rejects_recursive_provider_state(
    request_body: dict[str, Any],
    field_path: str,
    classification: str,
) -> None:
    result = sa.classify_session_owner_replay_safety_body(request_body)

    assert result.safe is False
    assert result.field_path == field_path
    assert result.classification == classification
    assert sa.is_replay_safe_session_owner_redispatch_body(request_body) is False


@pytest.mark.parametrize(
    "previous_response_id",
    [None, "", "   "],
    ids=["null", "empty", "whitespace"],
)
def test_replay_safety_treats_empty_previous_response_id_as_absent(
    previous_response_id: Any,
) -> None:
    body = {
        "previous_response_id": previous_response_id,
        "input": [{"role": "user", "content": "portable"}],
    }

    result = sa.classify_session_owner_replay_safety_body(body)

    assert result.safe is True
    assert sa.is_replay_safe_session_owner_redispatch_body(body) is True


def test_replay_unsafe_409_detail_omits_request_owner_and_candidate_secrets() -> None:
    replay_safety = sa.classify_session_owner_replay_safety_body(
        {
            "attacker-secret-ancestor": [
                {"previous_response_id": "resp-secret-provider-state"}
            ]
        }
    )
    guard = sa.SessionOwnerGuardResult(
        decision=sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
        session_identity="canonical-session-secret",
        cache_key="cache-key-secret",
        owner_id="owner-id-secret",
        owner_record={
            "state": "owned",
            "owner": "owner-id-secret",
            "attributes": {
                "provider": "owner-provider-secret",
                "model": "owner-model-secret",
                "route_family": "owner-route-secret",
                "account_hash": "owner-account-secret",
            },
        },
        mismatch_reason="mismatch-reason-secret",
    )
    message = (
        "Codex auto-review cannot own provider response state. Send a "
        "self-contained replay-safe body without previous_response_id "
        "or unsafe opaque rs_* provider state."
    )

    with pytest.raises(HTTPException) as exc_info:
        sa.raise_session_owner_redispatch_required(
            session_identity="canonical-session-secret",
            guard=guard,
            alias_model="codex-auto-review",
            candidate={
                "provider": "candidate-provider-secret",
                "model": "candidate-model-secret",
                "route_family": "candidate-route-secret",
                "codex_oauth_account_hash": "candidate-account-secret",
            },
            failure_phase="session_owner_replay_unsafe_auto_review",
            message=message,
            replay_safety=replay_safety,
        )

    assert exc_info.value.status_code == 409
    detail = exc_info.value.detail
    assert detail == {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "code": "aawm_session_owner_redispatch_required",
        },
        "redispatch_required": True,
        "redispatch_reason": "previous_response_id",
        "failure_phase": "session_owner_replay_unsafe_auto_review",
        "attempted_provider_call": False,
        "alias_model": "codex-auto-review",
        "redispatch_model": "codex-auto-review",
        "replay_safety": {
            "field_path": "$.*[0].previous_response_id",
            "classification": "previous_response_id",
        },
    }
    detail_text = json.dumps(detail, sort_keys=True)
    for secret in (
        "attacker-secret-ancestor",
        "resp-secret-provider-state",
        "canonical-session-secret",
        "cache-key-secret",
        "owner-id-secret",
        "owner-provider-secret",
        "owner-model-secret",
        "owner-route-secret",
        "owner-account-secret",
        "mismatch-reason-secret",
        "candidate-provider-secret",
        "candidate-model-secret",
        "candidate-route-secret",
        "candidate-account-secret",
    ):
        assert secret not in detail_text


@pytest.mark.asyncio
async def test_clear_compatible_nonheld_guard_accepts_portable_safe_body() -> None:
    redis = _FakeRedisCache()
    request = type("Req", (), {})()
    request.state = type("State", (), {})()
    session_identity = "sess-openai-027-rebind"
    current = _full_attrs(credential_affinity="interchangeable")
    alternate = _full_attrs(
        account_hash="acct-2",
        account_label="secondary",
        account_lane="lane-b",
        credential_affinity="interchangeable",
    )
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        lease = await _seed_nonheld_compatible_owner_lease(
            request=request,
            session_identity=session_identity,
            attributes=current,
        )
        before = _redis_snapshot(redis)
        cleared = (
            await sa.clear_compatible_non_held_request_session_owner_guard_for_failover(
                request=request,
                request_body={
                    "model": current["model"],
                    "input": [{"role": "user", "content": "portable"}],
                },
                current_attributes=current,
                alternate_attributes=alternate,
                account_failover_planned=True,
                account_failover_replay_safe=True,
                has_account_bound_state=False,
                post_commit_retry=False,
            )
        )
        after = _redis_snapshot(redis)

    assert cleared is True
    assert sa.get_request_session_owner_lease(request) is None
    assert request.state._aawm_session_owner_guarded is False
    assert before == after
    assert lease.released is False
    assert lease.promoted is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_body",
    [
        {"previous_response_id": "resp_1"},
        {"input": [{"previous_response_id": "resp_1"}]},
        {"input": [{"id": "rs_opaque_item"}]},
    ],
)
async def test_clear_compatible_nonheld_guard_rejects_provider_owned_replay_state(
    request_body: dict[str, Any],
) -> None:
    redis = _FakeRedisCache()
    request = type("Req", (), {})()
    request.state = type("State", (), {})()
    session_identity = "sess-openai-027-replay-reject"
    current = _full_attrs(credential_affinity="interchangeable")
    alternate = _full_attrs(
        account_hash="acct-2",
        account_label="secondary",
        account_lane="lane-b",
        credential_affinity="interchangeable",
    )
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        lease = await _seed_nonheld_compatible_owner_lease(
            request=request,
            session_identity=session_identity,
            attributes=current,
        )
        before = _redis_snapshot(redis)
        cleared = (
            await sa.clear_compatible_non_held_request_session_owner_guard_for_failover(
                request=request,
                request_body=request_body,
                current_attributes=current,
                alternate_attributes=alternate,
                account_failover_planned=True,
                account_failover_replay_safe=True,
                has_account_bound_state=False,
                post_commit_retry=False,
            )
        )
        after = _redis_snapshot(redis)

    assert cleared is False
    assert sa.get_request_session_owner_lease(request) is lease
    assert request.state._aawm_session_owner_guarded is True
    assert before == after


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failover_overrides", "credential_affinity"),
    [
        pytest.param(
            {"account_failover_planned": False},
            "interchangeable",
            id="not-planned",
        ),
        pytest.param(
            {"account_failover_replay_safe": False},
            "interchangeable",
            id="not-replay-safe",
        ),
        pytest.param(
            {"has_account_bound_state": True},
            "interchangeable",
            id="account-bound-state",
        ),
        pytest.param(
            {"post_commit_retry": True},
            "interchangeable",
            id="post-commit",
        ),
        pytest.param({"failover_ordinal": 2}, "interchangeable", id="second-move"),
        pytest.param({}, "account_bound", id="non-interchangeable"),
    ],
)
async def test_clear_compatible_nonheld_guard_rejects_unsafe_failover_context(
    failover_overrides: dict[str, Any],
    credential_affinity: str,
) -> None:
    redis = _FakeRedisCache()
    request = type("Req", (), {})()
    request.state = type("State", (), {})()
    session_identity = "sess-openai-027-context-reject"
    current = _full_attrs(credential_affinity=credential_affinity)
    alternate = _full_attrs(
        account_hash="acct-2",
        account_label="secondary",
        account_lane="lane-b",
        credential_affinity=credential_affinity,
    )
    failover_context = {
        "account_failover_planned": True,
        "account_failover_replay_safe": True,
        "has_account_bound_state": False,
        "post_commit_retry": False,
        "failover_ordinal": 1,
        **failover_overrides,
    }
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        lease = await _seed_nonheld_compatible_owner_lease(
            request=request,
            session_identity=session_identity,
            attributes=current,
        )
        before = _redis_snapshot(redis)
        cleared = (
            await sa.clear_compatible_non_held_request_session_owner_guard_for_failover(
                request=request,
                request_body={"input": [{"role": "user", "content": "portable"}]},
                current_attributes=current,
                alternate_attributes=alternate,
                **failover_context,
            )
        )
        after = _redis_snapshot(redis)

    assert cleared is False
    assert sa.get_request_session_owner_lease(request) is lease
    assert request.state._aawm_session_owner_guarded is True
    assert before == after


@pytest.mark.asyncio
async def test_clear_compatible_nonheld_guard_rejects_held_lease() -> None:
    redis = _FakeRedisCache()
    request = type("Req", (), {})()
    request.state = type("State", (), {})()
    session_identity = "sess-openai-027-held"
    current = _full_attrs(credential_affinity="interchangeable")
    alternate = _full_attrs(
        account_hash="acct-2",
        account_label="secondary",
        account_lane="lane-b",
        credential_affinity="interchangeable",
    )
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        reserved = await sa.guard_session_owner_before_egress(
            session_identity=session_identity,
            requested_attributes=current,
        )
        lease = sa.lease_from_guard_result(reserved, attributes=current)
        sa.set_request_session_owner_lease(request, lease)
        before = _redis_snapshot(redis)
        cleared = (
            await sa.clear_compatible_non_held_request_session_owner_guard_for_failover(
                request=request,
                request_body={"input": [{"role": "user", "content": "portable"}]},
                current_attributes=current,
                alternate_attributes=alternate,
                account_failover_planned=True,
                account_failover_replay_safe=True,
                has_account_bound_state=False,
                post_commit_retry=False,
            )
        )
        after = _redis_snapshot(redis)

    assert cleared is False
    assert sa.get_request_session_owner_lease(request) is lease
    assert request.state._aawm_session_owner_guarded is True
    assert before == after


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("durable_state", "durable_owner"),
    [("reserved", "foreign-owner"), ("owned", "foreign-owner")],
)
async def test_clear_compatible_nonheld_guard_rejects_durable_reservation_or_foreign_owner(
    durable_state: str,
    durable_owner: str,
) -> None:
    redis = _FakeRedisCache()
    request = type("Req", (), {})()
    request.state = type("State", (), {})()
    session_identity = "sess-openai-027-durable-reject"
    current = _full_attrs(credential_affinity="interchangeable")
    alternate = _full_attrs(
        account_hash="acct-2",
        account_label="secondary",
        account_lane="lane-b",
        credential_affinity="interchangeable",
    )
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        lease = await _seed_nonheld_compatible_owner_lease(
            request=request,
            session_identity=session_identity,
            attributes=current,
        )
        cache_key = sa.build_aawm_alias_routing_session_owner_cache_key(
            session_identity=session_identity
        )
        namespaced_key = redis.check_and_fix_namespace(cache_key)
        record = json.loads(redis._data[namespaced_key].decode("utf-8"))
        record["state"] = durable_state
        record["owner"] = durable_owner
        if durable_state == "reserved":
            record["reservation_token"] = "foreign-token"
        redis._data[namespaced_key] = json.dumps(record).encode("utf-8")
        before = _redis_snapshot(redis)
        cleared = (
            await sa.clear_compatible_non_held_request_session_owner_guard_for_failover(
                request=request,
                request_body={"input": [{"role": "user", "content": "portable"}]},
                current_attributes=current,
                alternate_attributes=alternate,
                account_failover_planned=True,
                account_failover_replay_safe=True,
                has_account_bound_state=False,
                post_commit_retry=False,
            )
        )
        after = _redis_snapshot(redis)

    assert cleared is False
    assert sa.get_request_session_owner_lease(request) is lease
    assert request.state._aawm_session_owner_guarded is True
    assert before == after


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "alternate_overrides",
    [
        {"provider": "xai"},
        {"model": "gpt-5.6-other"},
        {"route_family": "other_route"},
        {"endpoint_contract": "other_endpoint"},
        {"state_format": "other_state"},
    ],
)
async def test_clear_compatible_nonheld_guard_rejects_incompatible_alternate(
    alternate_overrides: dict[str, Any],
) -> None:
    redis = _FakeRedisCache()
    request = type("Req", (), {})()
    request.state = type("State", (), {})()
    session_identity = "sess-openai-027-incompatible"
    current = _full_attrs(credential_affinity="interchangeable")
    alternate = _full_attrs(
        account_hash="acct-2",
        account_label="secondary",
        account_lane="lane-b",
        credential_affinity="interchangeable",
        **alternate_overrides,
    )
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        lease = await _seed_nonheld_compatible_owner_lease(
            request=request,
            session_identity=session_identity,
            attributes=current,
        )
        before = _redis_snapshot(redis)
        cleared = (
            await sa.clear_compatible_non_held_request_session_owner_guard_for_failover(
                request=request,
                request_body={"input": [{"role": "user", "content": "portable"}]},
                current_attributes=current,
                alternate_attributes=alternate,
                account_failover_planned=True,
                account_failover_replay_safe=True,
                has_account_bound_state=False,
                post_commit_retry=False,
            )
        )
        after = _redis_snapshot(redis)

    assert cleared is False
    assert sa.get_request_session_owner_lease(request) is lease
    assert request.state._aawm_session_owner_guarded is True
    assert before == after


@pytest.mark.asyncio
@pytest.mark.parametrize("lease_state", ["promoted", "released"])
async def test_clear_compatible_nonheld_guard_rejects_promoted_or_released_lease(
    lease_state: str,
) -> None:
    redis = _FakeRedisCache()
    request = type("Req", (), {})()
    request.state = type("State", (), {})()
    session_identity = "sess-openai-027-lease-state"
    current = _full_attrs(credential_affinity="interchangeable")
    alternate = _full_attrs(
        account_hash="acct-2",
        account_label="secondary",
        account_lane="lane-b",
        credential_affinity="interchangeable",
    )
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        lease = await _seed_nonheld_compatible_owner_lease(
            request=request,
            session_identity=session_identity,
            attributes=current,
        )
        setattr(lease, lease_state, True)
        before = _redis_snapshot(redis)
        cleared = (
            await sa.clear_compatible_non_held_request_session_owner_guard_for_failover(
                request=request,
                request_body={"input": [{"role": "user", "content": "portable"}]},
                current_attributes=current,
                alternate_attributes=alternate,
                account_failover_planned=True,
                account_failover_replay_safe=True,
                has_account_bound_state=False,
                post_commit_retry=False,
            )
        )
        after = _redis_snapshot(redis)

    assert cleared is False
    assert sa.get_request_session_owner_lease(request) is lease
    assert request.state._aawm_session_owner_guarded is True
    assert before == after


def test_explicit_session_identity_is_authoritative() -> None:
    request = type("Request", (), {"headers": {"thread-id": "header-thread"}})()
    body = {
        "thread_id": "body-thread",
        "session_id": "body-session",
        "litellm_metadata": {
            "thread_id": "metadata-thread",
            "session_id": "metadata-session",
        },
    }

    assert (
        sa.resolve_canonical_session_identity(
            request=request,
            request_body=body,
            session_identity="alias:explicit-thread:lane",
        )
        == "explicit-thread"
    )


def test_session_identity_fallback_is_preserved() -> None:
    assert (
        sa.resolve_canonical_session_identity(
            request_body={
                "session_id": "body-session",
                "litellm_metadata": {"aawm_session_id": "metadata-session"},
            }
        )
        == "metadata-session"
    )


def test_child_threads_sharing_session_resolve_distinct_identities() -> None:
    first = sa.resolve_canonical_session_identity(
        request_body={"thread_id": "child-one", "session_id": "shared-session"}
    )
    second = sa.resolve_canonical_session_identity(
        request_body={"thread_id": "child-two", "session_id": "shared-session"}
    )

    assert first == "child-one"
    assert second == "child-two"
    assert first != second


def test_same_child_thread_identity_is_stable_across_sources() -> None:
    metadata_identity = sa.resolve_canonical_session_identity(
        request_body={
            "litellm_metadata": {"aawm_thread_id": "stable-child"},
            "session_id": "shared-session",
        }
    )
    request = type(
        "Request",
        (),
        {"headers": {"x-aawm-thread-id": "stable-child"}},
    )()
    header_identity = sa.resolve_canonical_session_identity(
        request=request,
        request_body={"session_id": "shared-session"},
    )

    assert metadata_identity == header_identity == "stable-child"


def test_review_thread_is_preferred_over_session_identity() -> None:
    assert (
        sa.resolve_canonical_session_identity(
            request_body={
                "claude_thread_id": "review-thread",
                "session_id": "shared-session",
            }
        )
        == "review-thread"
    )


def test_parent_thread_identity_is_ignored() -> None:
    request = type(
        "Request",
        (),
        {"headers": {"x-codex-parent-thread-id": "header-parent"}},
    )()
    assert (
        sa.resolve_canonical_session_identity(
            request=request,
            request_body={
                "parent_thread_id": "body-parent",
                "session_id": "session-fallback",
                "litellm_metadata": {"parent_thread_id": "metadata-parent"},
            },
        )
        == "session-fallback"
    )


def test_optional_dispatch_labels_are_ignored() -> None:
    request = type(
        "Request",
        (),
        {"headers": {"x-openai-subagent": "review-agent"}},
    )()
    assert (
        sa.resolve_canonical_session_identity(
            request=request,
            request_body={
                "agent_id": "agent-1",
                "dispatch_id": "dispatch-1",
                "thread_source": "subagent",
                "litellm_metadata": {
                    "agent_id": "agent-2",
                    "dispatch_id": "dispatch-2",
                    "thread_source": "subagent",
                },
            },
        )
        is None
    )


def test_exact_thread_id_header_spelling_is_supported() -> None:
    request = type(
        "Request",
        (),
        {
            "headers": {
                "thread-id": "exact-header-thread",
                "x-session-id": "session-fallback",
            }
        },
    )()

    assert (
        sa.resolve_canonical_session_identity(request=request)
        == "exact-header-thread"
    )


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
async def test_foreign_reservation_release_then_selector_and_guard_reserve() -> None:
    redis = _FakeRedisCache()
    attrs = _full_attrs()
    request = type("Req", (), {})()
    request.state = type("State", (), {})()
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        leader = await sa.guard_session_owner_before_egress(
            session_identity="sess-wait-release",
            requested_attributes=attrs,
        )
        release_once = True

        async def release_during_wait(_: float) -> None:
            nonlocal release_once
            if release_once:
                release_once = False
                released = await sa.release_session_owner_reservation(
                    session_identity="sess-wait-release",
                    reservation_token=leader.reservation_token,
                )
                assert released.outcome is sa.SessionOwnerMutationOutcome.RELEASED

        with patch.object(sa.asyncio, "sleep", side_effect=release_during_wait):
            record, _, error = await sa.get_session_owner_record(
                session_identity="sess-wait-release",
                request=request,
                wait_for_foreign_reservation=True,
                reservation_wait_timeout_seconds=0.25,
                reservation_wait_poll_seconds=0.001,
            )
        assert record is None
        assert error is None

        follower = await sa.guard_session_owner_before_egress(
            session_identity="sess-wait-release",
            request=request,
            requested_attributes=attrs,
        )

    assert follower.decision is sa.SessionOwnerGuardDecision.UNOWNED_RESERVED
    assert follower.held_reservation is True
    assert follower.reservation_token != leader.reservation_token


@pytest.mark.asyncio
async def test_foreign_reservation_release_during_guard_wait_reserves() -> None:
    redis = _FakeRedisCache()
    attrs = _full_attrs()
    request = type("Req", (), {})()
    request.state = type("State", (), {})()
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        leader = await sa.guard_session_owner_before_egress(
            session_identity="sess-guard-wait-release",
            requested_attributes=attrs,
        )
        release_once = True

        async def release_during_wait(_: float) -> None:
            nonlocal release_once
            if release_once:
                release_once = False
                released = await sa.release_session_owner_reservation(
                    session_identity="sess-guard-wait-release",
                    reservation_token=leader.reservation_token,
                )
                assert released.outcome is sa.SessionOwnerMutationOutcome.RELEASED

        with patch.object(sa.asyncio, "sleep", side_effect=release_during_wait):
            follower = await sa.guard_session_owner_before_egress(
                session_identity="sess-guard-wait-release",
                request=request,
                requested_attributes=attrs,
                reservation_wait_timeout_seconds=0.25,
                reservation_wait_poll_seconds=0.001,
            )

    assert follower.decision is sa.SessionOwnerGuardDecision.UNOWNED_RESERVED
    assert follower.held_reservation is True
    assert follower.reservation_token != leader.reservation_token


@pytest.mark.asyncio
async def test_get_then_guard_shares_expired_request_wait_deadline() -> None:
    redis = _FakeRedisCache()
    attrs = _full_attrs()
    request = type("Req", (), {})()
    request.state = type("State", (), {})()
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        leader = await sa.guard_session_owner_before_egress(
            session_identity="sess-shared-wait-deadline",
            requested_attributes=attrs,
        )
        sleep = AsyncMock()
        with patch.object(sa.asyncio, "sleep", new=sleep):
            record, _, error = await sa.get_session_owner_record(
                session_identity="sess-shared-wait-deadline",
                request=request,
                wait_for_foreign_reservation=True,
                reservation_wait_timeout_seconds=0.0,
                reservation_wait_poll_seconds=0.001,
            )
            follower = await sa.guard_session_owner_before_egress(
                session_identity="sess-shared-wait-deadline",
                request=request,
                requested_attributes=attrs,
                reservation_wait_timeout_seconds=0.25,
                reservation_wait_poll_seconds=0.001,
            )

    assert record is not None
    assert record.get("reservation_token") == leader.reservation_token
    assert error is None
    assert follower.decision is sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED
    assert "concurrent reservation" in (follower.mismatch_reason or "")
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_foreign_reservation_promotion_compatible_owner_proceeds() -> None:
    redis = _FakeRedisCache()
    attrs = _full_attrs()
    request = type("Req", (), {})()
    request.state = type("State", (), {})()
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        leader = await sa.guard_session_owner_before_egress(
            session_identity="sess-wait-compatible",
            requested_attributes=attrs,
        )
        promote_once = True

        async def promote_during_wait(_: float) -> None:
            nonlocal promote_once
            if promote_once:
                promote_once = False
                promoted = await sa.promote_session_owner_reservation(
                    session_identity="sess-wait-compatible",
                    reservation_token=leader.reservation_token,
                    attributes=attrs,
                )
                assert promoted.outcome is sa.SessionOwnerMutationOutcome.PROMOTED

        with patch.object(sa.asyncio, "sleep", side_effect=promote_during_wait):
            record, _, error = await sa.get_session_owner_record(
                session_identity="sess-wait-compatible",
                request=request,
                wait_for_foreign_reservation=True,
                reservation_wait_timeout_seconds=0.25,
                reservation_wait_poll_seconds=0.001,
            )
        assert sa._record_state(record) == "owned"
        assert error is None

        follower = await sa.guard_session_owner_before_egress(
            session_identity="sess-wait-compatible",
            request=request,
            requested_attributes=attrs,
        )

    assert follower.decision is sa.SessionOwnerGuardDecision.COMPATIBLE_OWNER
    assert follower.held_reservation is False
    assert follower.owner_id == leader.owner_id


@pytest.mark.asyncio
async def test_foreign_reservation_same_hosted_openai_model_hop_is_compatible_before_egress() -> None:
    redis = _FakeRedisCache()
    attrs = _full_attrs(model="model-a")
    hop = _full_attrs(model="model-b")
    request = type("Req", (), {})()
    request.state = type("State", (), {})()
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        leader = await sa.guard_session_owner_before_egress(
            session_identity="sess-wait-model-hop",
            requested_attributes=hop,
        )
        promote_once = True

        async def promote_during_wait(_: float) -> None:
            nonlocal promote_once
            if promote_once:
                promote_once = False
                promoted = await sa.promote_session_owner_reservation(
                    session_identity="sess-wait-model-hop",
                    reservation_token=leader.reservation_token,
                    attributes=hop,
                )
                assert promoted.outcome is sa.SessionOwnerMutationOutcome.PROMOTED

        with patch.object(sa.asyncio, "sleep", side_effect=promote_during_wait):
            record, _, error = await sa.get_session_owner_record(
                session_identity="sess-wait-model-hop",
                request=request,
                wait_for_foreign_reservation=True,
                reservation_wait_timeout_seconds=0.25,
                reservation_wait_poll_seconds=0.001,
            )
        assert sa._record_state(record) == "owned"
        assert error is None

        guard = await sa.guard_session_owner_before_egress(
            session_identity="sess-wait-model-hop",
            request=request,
            requested_attributes=attrs,
        )

    assert guard.decision is sa.SessionOwnerGuardDecision.COMPATIBLE_OWNER
    assert guard.held_reservation is False
    assert guard.owner_id == leader.owner_id
    assert guard.mismatch_reason is None


@pytest.mark.asyncio
async def test_foreign_reservation_timeout_preserves_existing_409() -> None:
    redis = _FakeRedisCache()
    attrs = _full_attrs()
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        leader = await sa.guard_session_owner_before_egress(
            session_identity="sess-wait-timeout",
            requested_attributes=attrs,
        )
        follower = await sa.guard_session_owner_before_egress(
            session_identity="sess-wait-timeout",
            requested_attributes=attrs,
            reservation_wait_timeout_seconds=0.0,
            reservation_wait_poll_seconds=0.001,
        )

    assert follower.decision is sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED
    assert follower.owner_record is not None
    assert follower.owner_record.get("reservation_token") == leader.reservation_token
    assert "concurrent reservation" in (follower.mismatch_reason or "")


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
async def test_same_hosted_openai_account_hop_is_compatible_before_egress() -> None:
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
        hop = dict(attrs)
        hop["account_hash"] = "acct-other"
        g2 = await sa.guard_session_owner_before_egress(
            session_identity="sess-acct", requested_attributes=hop
        )
        assert g2.decision is sa.SessionOwnerGuardDecision.COMPATIBLE_OWNER
        assert g2.held_reservation is False
        assert g2.owner_id == g.owner_id
        assert g2.mismatch_reason is None


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
async def test_real_redis_session_owner_failure_raises_409_before_openai_egress() -> None:
    from types import SimpleNamespace

    from litellm.proxy.pass_through_endpoints import pass_through_endpoints as pte

    async_redis_client = AsyncMock()
    async_redis_client.get.side_effect = redis.exceptions.ConnectionError(
        "redis unavailable"
    )
    with patch(
        "litellm._redis.get_redis_client",
        return_value=MagicMock(),
    ), patch(
        "litellm._redis.get_redis_connection_pool",
        return_value=MagicMock(),
    ), patch.object(
        RedisCache,
        "_setup_health_pings",
    ):
        redis_cache = RedisCache()

    dual_cache = DualCache(
        in_memory_cache=InMemoryCache(),
        redis_cache=redis_cache,
    )
    session_id = "sess-real-redis-down"
    body = {
        "model": "gpt-5.6-sol",
        "litellm_metadata": {"session_id": session_id},
    }
    request = SimpleNamespace(
        state=SimpleNamespace(),
        method="POST",
        headers={"authorization": "Bearer redacted"},
        url=httpx.URL("https://api.openai.com/v1/responses"),
    )

    with patch.object(
        durable_mod,
        "get_aawm_alias_routing_dual_cache",
        return_value=dual_cache,
    ), patch.object(
        durable_mod,
        "get_aawm_alias_routing_state_namespace",
        return_value="ns",
    ), patch.object(
        redis_cache,
        "init_async_client",
        return_value=async_redis_client,
    ), pytest.raises(HTTPException) as exc_info:
        await pte._aawm_session_owner_pre_send_guard(
            request=request,
            parsed_body=body,
            custom_llm_provider="openai",
            egress_credential_family="openai",
            expected_target_family="openai",
            url=request.url,
        )

    assert async_redis_client.get.await_count == 1
    async_redis_client.set.assert_not_awaited()
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["error"]["code"] == (
        "aawm_session_owner_redispatch_required"
    )
    assert exc_info.value.detail["redispatch_required"] is True
    assert exc_info.value.detail["attempted_provider_call"] is False


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
async def test_same_hosted_openai_model_hop_is_compatible_owner() -> None:
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
        # Same hosted OpenAI provider + different model remains compatible.
        hop = dict(owned)
        hop["model"] = "model-b"
        g2 = await sa.guard_session_owner_before_egress(
            session_identity="sess-exact", requested_attributes=hop
        )
    assert g2.decision is sa.SessionOwnerGuardDecision.COMPATIBLE_OWNER
    assert g2.held_reservation is False
    assert g2.owner_id == g.owner_id
    assert g2.mismatch_reason is None


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
async def test_sequential_candidates_replace_released_request_lease() -> None:
    redis = _FakeRedisCache()
    attrs = _full_attrs()
    request = type("Req", (), {})()
    request.state = type("State", (), {})()
    session_identity = "sess-sequential-candidates"

    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        first_guard = await sa.ensure_session_owner_guard_for_request(
            request=request,
            session_identity=session_identity,
            requested_attributes=attrs,
        )
        first_lease = sa.get_request_session_owner_lease(request)
        assert first_guard.held_reservation is True
        assert first_lease is not None
        first_token = first_lease.reservation_token

        first_result = await sa.finalize_request_session_owner_lease(
            request, exc=RuntimeError("first candidate failed")
        )
        assert first_result is not None
        assert first_result.outcome is sa.SessionOwnerMutationOutcome.RELEASED
        assert first_lease.released is True

        second_guard = await sa.ensure_session_owner_guard_for_request(
            request=request,
            session_identity=session_identity,
            requested_attributes=attrs,
        )
        second_lease = sa.get_request_session_owner_lease(request)
        assert second_guard.held_reservation is True
        assert second_lease is not None
        assert second_lease is not first_lease
        assert second_lease.reservation_token != first_token
        assert second_lease.promoted is False
        assert second_lease.released is False

        second_result = await sa.finalize_request_session_owner_lease(
            request, exc=RuntimeError("second candidate failed")
        )
        assert second_result is not None
        assert second_result.outcome is sa.SessionOwnerMutationOutcome.RELEASED
        assert second_lease.released is True

        third_guard = await sa.guard_session_owner_before_egress(
            session_identity=session_identity,
            requested_attributes=attrs,
        )
        assert third_guard.decision is sa.SessionOwnerGuardDecision.UNOWNED_RESERVED


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


def test_effective_redispatch_identity_is_deterministic_server_only_and_single_generation() -> None:
    from starlette.datastructures import State

    first_request = type("Req", (), {})()
    first_request.state = State()
    first_request.headers = {"x-thread-id": "untrusted-thread-a", "x-test": "one"}
    second_request = type("Req", (), {})()
    second_request.state = State()
    second_request.headers = {"x-thread-id": "untrusted-thread-b", "x-test": "two"}
    headers_before = dict(first_request.headers)
    base_identity = "canonical-base-thread"

    first = sa.activate_session_owner_redispatch_effective_identity(
        request=first_request,
        base_session_identity=base_identity,
    )
    second = sa.activate_session_owner_redispatch_effective_identity(
        request=second_request,
        base_session_identity=base_identity,
    )

    expected = (
        f"{sa._SESSION_OWNER_REDISPATCH_EFFECTIVE_IDENTITY_PREFIX}"
        f"{hashlib.sha256((sa._SESSION_OWNER_REDISPATCH_EFFECTIVE_IDENTITY_DOMAIN_SEPARATOR + base_identity).encode('utf-8')).hexdigest()}"
    )
    assert first == expected
    assert second == expected
    assert first_request.headers == headers_before
    assert (
        sa.resolve_canonical_session_identity(
            first_request,
            {"agent_id": "untrusted-agent", "thread_id": "other-thread"},
        )
        == expected
    )
    assert (
        sa.activate_session_owner_redispatch_effective_identity(
            request=first_request,
            base_session_identity=expected,
        )
        is None
    )
    assert sa.get_request_effective_session_identity(first_request) == expected
    mock_request = type("Req", (), {})()
    mock_request.state = MagicMock()
    assert sa.get_request_effective_session_identity(mock_request) is None


def test_codex_auto_review_identity_is_exact_idempotent_and_canonical() -> None:
    from starlette.datastructures import State

    request = type("Req", (), {})()
    request.state = State()
    request.headers = {"x-thread-id": " parent-thread "}
    parent_identity = sa.resolve_canonical_session_identity(request=request)

    first = sa.activate_codex_auto_review_session_identity(
        request=request,
        parent_session_identity=parent_identity,
    )
    second = sa.activate_codex_auto_review_session_identity(
        request=request,
        parent_session_identity=first,
    )
    already_suffixed_request = type("Req", (), {})()
    already_suffixed_request.state = State()
    already_suffixed = sa.activate_codex_auto_review_session_identity(
        request=already_suffixed_request,
        parent_session_identity=first,
    )

    assert first == second == already_suffixed == "parent-thread:codex-auto-review"
    assert sa.get_request_codex_auto_review_session_identity(request) == first
    assert (
        sa.get_request_codex_auto_review_parent_session_identity(request)
        == "parent-thread"
    )
    request_call = request.state.aawm_alias_request_litellm_call_id = "request-call"
    owner_identity = sa.activate_codex_auto_review_session_owner_identity(
        request=request,
        parent_session_identity="parent-thread",
        request_call_identity=request_call,
    )
    assert owner_identity.startswith("aawm-codex-auto-review-owner-v1:")
    assert owner_identity != "parent-thread"
    assert owner_identity != "parent-thread:codex-auto-review"
    assert sa.get_request_effective_session_identity(request) == owner_identity
    assert sa.resolve_canonical_session_identity(request=request) == owner_identity
    assert sa.get_request_codex_auto_review_parent_session_identity(request) == (
        "parent-thread"
    )
    assert (
        sa.get_request_codex_auto_review_parent_session_identity(
                already_suffixed_request
            )
            == "parent-thread"
        )
    request.state._state.pop(
        sa._REQUEST_STATE_EFFECTIVE_SESSION_IDENTITY_ATTR,
        None,
    )
    assert (
        sa.resolve_canonical_session_identity(
            request=request,
            request_body={"thread_id": "body-thread"},
        )
        == first
    )
    assert sa.request_has_effective_session_identity(request) is False
    effective_identity = sa.activate_session_owner_redispatch_effective_identity(
        request=request,
        base_session_identity=first,
    )
    assert effective_identity is not None
    assert (
        sa.resolve_canonical_session_identity(
            request=request,
            request_body={"thread_id": "body-thread"},
        )
        == effective_identity
    )
    assert (
        sa.get_request_codex_auto_review_parent_session_identity(request)
        == "parent-thread"
    )


def _codex_selector_request(thread_id: str = "selector-thread") -> Any:
    request = type("Req", (), {})()
    request.state = type("State", (), {})()
    request.headers = {"x-thread-id": thread_id}
    return request


def _patch_codex_selector_basics(
    monkeypatch: pytest.MonkeyPatch,
    *,
    has_account_bound_state: bool = True,
) -> tuple[Any, dict[str, Any]]:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import selection as sel

    selector_globals = sel._select_codex_auto_agent_candidate.__globals__
    selected_candidate = {
        "provider": "openai",
        "model": "gpt-5.4-codex",
        "route_family": "codex_oauth",
    }
    selected_state = {
        "candidate": selected_candidate,
        "cooldown_seconds": 0.0,
        "failover_ordinal": 0,
    }
    monkeypatch.setitem(
        selector_globals,
        "_lookup_active_snapshot_canonical_alias",
        lambda *_args, **_kwargs: "alias",
    )
    monkeypatch.setitem(selector_globals, "_extract_client_product_label", lambda *_args: None)
    monkeypatch.setitem(
        selector_globals, "_resolve_codex_session_key", lambda *_args, **_kwargs: "key"
    )
    monkeypatch.setitem(selector_globals, "_has_continuation_state", lambda _body: True)

    def _has_account_bound_state(_body: Any) -> bool:
        return has_account_bound_state

    monkeypatch.setitem(
        selector_globals, "_has_account_bound_state", _has_account_bound_state
    )
    monkeypatch.setattr(sel, "_has_account_bound_state", _has_account_bound_state)
    monkeypatch.setitem(
        selector_globals, "_get_codex_session_affinity", AsyncMock(return_value=None)
    )
    monkeypatch.setitem(
        selector_globals,
        "_apply_codex_oauth_inventory_affinity_policy",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setitem(
        selector_globals, "_build_auto_agent_skipped_candidates_from_states", lambda _states: []
    )
    monkeypatch.setitem(
        selector_globals,
        "_attach_aawm_alias_routing_state_sources",
        lambda selection, **_kwargs: selection,
    )
    monkeypatch.setitem(
        selector_globals,
        "_build_codex_auto_agent_candidate_states",
        AsyncMock(return_value=[selected_state]),
    )
    monkeypatch.setitem(
        selector_globals,
        "_select_available_state",
        lambda _request, states, **_kwargs: states[0] if states else None,
    )
    return sel, selected_candidate


@pytest.mark.asyncio
@pytest.mark.parametrize("review_alias", ["codex-auto-review", "auto-review"])
async def test_auto_review_selector_uses_review_owner_identity_without_mutating_body(
    monkeypatch: pytest.MonkeyPatch,
    review_alias: str,
) -> None:
    sel, candidate = _patch_codex_selector_basics(monkeypatch)
    monkeypatch.setitem(
        sel._select_codex_auto_agent_candidate.__globals__,
        "_lookup_active_snapshot_canonical_alias",
        lambda *_args, **_kwargs: review_alias,
    )
    owner_lookup = AsyncMock(return_value=(None, "review-owner-key", None))
    monkeypatch.setattr(sa, "get_session_owner_record", owner_lookup)
    replay_classifier = MagicMock(
        wraps=sa.classify_session_owner_replay_safety_body
    )
    monkeypatch.setattr(
        sa,
        "classify_session_owner_replay_safety_body",
        replay_classifier,
    )
    request = _codex_selector_request("parent-thread")
    request_body = {
        "model": review_alias,
        "input": [
            {
                "type": "function_call",
                "name": "inspect",
                "call_id": "call-review",
            }
        ],
        "litellm_metadata": {
            "agent_id": "review-agent",
            "review": {"mode": "auto"},
        },
    }
    body_before = json.loads(json.dumps(request_body))
    headers_before = dict(request.headers)
    request.state.aawm_alias_request_litellm_call_id = "request-call-one"

    selected = await sel._select_codex_auto_agent_candidate(
        request=request,
        request_body=request_body,
    )

    review_identity = sa.get_request_effective_session_identity(request)
    owner_lookup.assert_awaited_once_with(
        session_identity=review_identity,
        request=request,
        wait_for_foreign_reservation=True,
    )
    assert selected["candidate"] == candidate
    assert selected["alias_model"] == review_alias
    assert selected["canonical_session_identity"] == "parent-thread"
    assert selected["session_owner_identity"] == review_identity
    assert review_identity is not None
    assert review_identity.startswith("aawm-codex-auto-review-owner-v1:")
    assert review_identity != "parent-thread"
    assert sa.get_request_codex_auto_review_session_identity(request) == (
        "parent-thread:codex-auto-review"
    )
    assert (
        sa.get_request_codex_auto_review_parent_session_identity(request)
        == "parent-thread"
    )
    assert request_body == body_before
    assert request.headers == headers_before
    replay_classifier.assert_called_once_with(request_body)


@pytest.mark.asyncio
@pytest.mark.parametrize("review_alias", ["codex-auto-review", "auto-review"])
async def test_auto_review_replays_full_adapter_reasoning_without_poisoning_guardian(
    monkeypatch: pytest.MonkeyPatch,
    review_alias: str,
) -> None:
    sel, candidate = _patch_codex_selector_basics(monkeypatch)
    selector_globals = sel._select_codex_auto_agent_candidate.__globals__
    monkeypatch.setitem(
        selector_globals,
        "_lookup_active_snapshot_canonical_alias",
        lambda *_args, **_kwargs: review_alias,
    )
    owner_lookup = AsyncMock(return_value=(None, "review-owner-key", None))
    monkeypatch.setattr(sa, "get_session_owner_record", owner_lookup)

    first_request = _codex_selector_request("guardian-thread")
    first_request.state.aawm_alias_request_litellm_call_id = "review-call-one"
    first_body = {
        "model": review_alias,
        "input": [{"role": "user", "content": "first review"}],
    }
    first_body_before = json.loads(json.dumps(first_body))
    first_headers_before = dict(first_request.headers)

    first_selection = await sel._select_codex_auto_agent_candidate(
        request=first_request,
        request_body=first_body,
    )

    adapter_item = _full_adapter_reasoning_item()
    second_request = _codex_selector_request("guardian-thread")
    second_request.state.aawm_alias_request_litellm_call_id = "review-call-two"
    second_body = {
        "model": review_alias,
        "input": [
            adapter_item,
            {"role": "user", "content": "second review"},
        ],
    }
    second_body_before = json.loads(json.dumps(second_body))
    second_headers_before = dict(second_request.headers)

    second_selection = await sel._select_codex_auto_agent_candidate(
        request=second_request,
        request_body=second_body,
    )

    first_owner = first_selection["session_owner_identity"]
    second_owner = second_selection["session_owner_identity"]
    assert first_selection["candidate"] == candidate
    assert second_selection["candidate"] == candidate
    assert first_owner.startswith("aawm-codex-auto-review-owner-v1:")
    assert second_owner.startswith("aawm-codex-auto-review-owner-v1:")
    assert first_owner != second_owner
    assert owner_lookup.await_count == 2
    assert [
        call.kwargs["session_identity"] for call in owner_lookup.await_args_list
    ] == [first_owner, second_owner]
    assert first_body == first_body_before
    assert first_request.headers == first_headers_before
    assert second_body == second_body_before
    assert second_request.headers == second_headers_before


@pytest.mark.asyncio
async def test_namespaced_codex_auto_review_lookup_uses_canonical_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sel, candidate = _patch_codex_selector_basics(monkeypatch)
    alias_lookup = MagicMock(return_value="codex-auto-review")
    monkeypatch.setitem(
        sel._select_codex_auto_agent_candidate.__globals__,
        "_lookup_active_snapshot_canonical_alias",
        alias_lookup,
    )
    monkeypatch.setattr(
        sa,
        "get_session_owner_record",
        AsyncMock(return_value=(None, "review-owner-key", None)),
    )
    request = _codex_selector_request("namespaced-parent")
    request.state.aawm_alias_request_litellm_call_id = "request-call-two"
    request_body = {
        "model": "  ChatGPT/Codex-Auto-Review  ",
        "input": [{"type": "function_call", "name": "inspect"}],
    }
    body_before = json.loads(json.dumps(request_body))

    selected = await sel._select_codex_auto_agent_candidate(
        request=request,
        request_body=request_body,
    )

    alias_lookup.assert_called_once_with("codex-auto-review", request=request)
    assert request_body == body_before
    assert selected["candidate"] == candidate
    assert selected["alias_model"] == "codex-auto-review"
    assert selected["canonical_session_identity"] == "namespaced-parent"
    assert selected["session_owner_identity"].startswith(
        "aawm-codex-auto-review-owner-v1:"
    )
    assert selected["session_owner_identity"] == (
        sa.get_request_effective_session_identity(request)
    )


@pytest.mark.asyncio
async def test_codex_compatible_owned_redispatch_metadata_remains_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sel, candidate = _patch_codex_selector_basics(monkeypatch)
    selector_globals = sel._select_codex_auto_agent_candidate.__globals__
    request = _codex_selector_request()
    owner_record = {
        "state": "owned",
        "owner": "owner-a",
        "attributes": dict(candidate),
    }
    monkeypatch.setattr(
        sa,
        "get_session_owner_record",
        AsyncMock(return_value=(owner_record, "owner-key", None)),
    )
    monkeypatch.setattr(
        sa,
        "owner_record_as_affinity_hint",
        lambda _record, **_kwargs: dict(candidate),
    )
    monkeypatch.setitem(
        selector_globals,
        "_find_codex_auto_agent_affinity_candidate",
        lambda *_args, **_kwargs: dict(candidate),
    )
    monkeypatch.setitem(
        selector_globals,
        "_build_codex_auto_agent_affinity_candidate_state",
        AsyncMock(return_value={"candidate": dict(candidate), "cooldown_seconds": 0.0}),
    )
    monkeypatch.setitem(selector_globals, "_candidate_matches_affinity", lambda *_args: True)
    monkeypatch.setitem(
        selector_globals,
        "_is_auto_agent_candidate_state_available",
        lambda _state: True,
    )
    activate = MagicMock(wraps=sa.activate_session_owner_redispatch_effective_identity)
    monkeypatch.setattr(sa, "activate_session_owner_redispatch_effective_identity", activate)

    selected = await sel._select_codex_auto_agent_candidate(
        request=request,
        request_body={
            "model": "alias",
            "litellm_metadata": {"redispatch_ordinal": 1},
            "input": [{"type": "function_call", "name": "inspect"}],
        },
    )

    assert selected["selection_reason"] == "session_affinity"
    assert selected["request_mode"] == "fresh_redispatch"
    assert selected["candidate"] == candidate
    activate.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("review_alias", ["codex-auto-review", "auto-review"])
async def test_auto_review_replay_unsafe_opaque_state_stays_409(
    monkeypatch: pytest.MonkeyPatch,
    review_alias: str,
) -> None:
    sel, _ = _patch_codex_selector_basics(
        monkeypatch,
        has_account_bound_state=True,
    )
    selector_globals = sel._select_codex_auto_agent_candidate.__globals__
    monkeypatch.setitem(
        selector_globals,
        "_lookup_active_snapshot_canonical_alias",
        lambda *_args, **_kwargs: review_alias,
    )
    request = _codex_selector_request("base-thread")
    request.state.aawm_alias_request_litellm_call_id = "unsafe-review-call"
    owner_record_lookup = AsyncMock(return_value=(None, "owner-key", None))
    monkeypatch.setattr(sa, "get_session_owner_record", owner_record_lookup)

    with pytest.raises(HTTPException) as exc_info:
        await sel._select_codex_auto_agent_candidate(
            request=request,
            request_body={
                "model": review_alias,
                "litellm_metadata": {"redispatch_ordinal": 1},
                "input": [
                    {
                        "type": "reasoning",
                        "id": "provider-item",
                        "encrypted_content": "ciphertext-must-not-leak",
                        "content": "reasoning-content-must-not-leak",
                    },
                    {"type": "reasoning", "id": "rs_sensitive_item"},
                ],
            },
        )

    assert exc_info.value.status_code == 409
    detail = exc_info.value.detail
    assert detail["error"]["code"] == "aawm_session_owner_redispatch_required"
    assert detail["failure_phase"] == "session_owner_replay_unsafe_auto_review"
    assert detail["redispatch_required"] is True
    assert detail["attempted_provider_call"] is False
    assert detail["replay_safety"] == {
        "field_path": "$.*[1].id",
        "classification": "id_only_reasoning_reference",
    }
    detail_text = str(detail)
    assert "ciphertext-must-not-leak" not in detail_text
    assert "reasoning-content-must-not-leak" not in detail_text
    assert "rs_sensitive_item" not in detail_text
    owner_record_lookup.assert_not_awaited()
    assert sa.get_request_effective_session_identity(request) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("review_alias", ["codex-auto-review", "auto-review"])
async def test_auto_review_full_adapter_owned_route_mismatch_uses_effective_identity(
    monkeypatch: pytest.MonkeyPatch,
    review_alias: str,
) -> None:
    sel, candidate = _patch_codex_selector_basics(monkeypatch)
    selector_globals = sel._select_codex_auto_agent_candidate.__globals__
    monkeypatch.setitem(
        selector_globals,
        "_lookup_active_snapshot_canonical_alias",
        lambda *_args, **_kwargs: review_alias,
    )
    request = _codex_selector_request("base-thread")
    request.state.aawm_alias_request_litellm_call_id = "owned-review-call"
    owner_record = {
        "state": "owned",
        "owner": "owner-a",
        "attributes": {
            "provider": "other",
            "model": "removed-model",
            "route_family": "other_route",
            "codex_oauth_account_label": "owner-account",
            "codex_oauth_account_hash": "owner-hash",
            "codex_oauth_lane_key": "owner-lane",
        },
    }
    owner_lookups: list[str] = []

    async def _get_owner_record(*, session_identity: str | None, **_kwargs: Any) -> tuple[Any, str, None]:
        assert session_identity is not None
        owner_lookups.append(session_identity)
        if len(owner_lookups) == 1:
            assert session_identity.startswith(
                "aawm-codex-auto-review-owner-v1:"
            )
            return owner_record, "owner-key", None
        assert session_identity.startswith(
            "aawm-session-owner-redispatch-v1:"
        )
        return None, "effective-key", None

    monkeypatch.setattr(sa, "get_session_owner_record", _get_owner_record)
    monkeypatch.setattr(
        sa,
        "owner_record_as_affinity_hint",
        lambda _record, **_kwargs: dict(owner_record["attributes"]),
    )
    monkeypatch.setitem(
        selector_globals,
        "_find_codex_auto_agent_affinity_candidate",
        lambda *_args, **_kwargs: None,
    )
    replay_classifier = MagicMock(
        wraps=sa.classify_session_owner_replay_safety_body
    )
    monkeypatch.setattr(
        sa,
        "classify_session_owner_replay_safety_body",
        replay_classifier,
    )
    request_body = {
        "model": review_alias,
        "litellm_metadata": {"redispatch_ordinal": 1},
        "input": [
            _full_adapter_reasoning_item(),
            {"role": "user", "content": "review this change"},
        ],
    }
    body_before = json.loads(json.dumps(request_body))
    headers_before = dict(request.headers)

    selected = await sel._select_codex_auto_agent_candidate(
        request=request,
        request_body=request_body,
    )

    assert selected["selection_reason"] == "first_available"
    assert selected["candidate"] == candidate
    assert selected["has_account_bound_state"] is True
    assert selected["canonical_session_identity"] == "base-thread"
    assert selected["session_owner_identity"].startswith(
        "aawm-session-owner-redispatch-v1:"
    )
    assert len(owner_lookups) == 2
    assert owner_lookups[1] == selected["session_owner_identity"]
    assert owner_lookups[0] != owner_lookups[1]
    assert sa.get_request_effective_session_identity(request) == selected[
        "session_owner_identity"
    ]
    assert (
        sa.get_request_codex_auto_review_parent_session_identity(request)
        == "base-thread"
    )
    assert selected["affinity_bypassed"] is True
    replay_classifier.assert_called_once_with(request_body)
    assert request_body == body_before
    assert request.headers == headers_before


@pytest.mark.asyncio
@pytest.mark.parametrize("review_alias", ["codex-auto-review", "auto-review"])
async def test_auto_review_replay_unsafe_previous_response_id_stays_409(
    monkeypatch: pytest.MonkeyPatch,
    review_alias: str,
) -> None:
    sel, _candidate = _patch_codex_selector_basics(monkeypatch)
    selector_globals = sel._select_codex_auto_agent_candidate.__globals__
    request = _codex_selector_request()
    monkeypatch.setitem(
        selector_globals,
        "_lookup_active_snapshot_canonical_alias",
        lambda *_args, **_kwargs: review_alias,
    )
    owner_record_lookup = AsyncMock()
    monkeypatch.setattr(sa, "get_session_owner_record", owner_record_lookup)
    monkeypatch.setattr(
        sa,
        "raise_session_owner_redispatch_required",
        lambda **kwargs: (_ for _ in ()).throw(HTTPException(status_code=409, detail=kwargs)),
    )

    with pytest.raises(HTTPException) as exc_info:
        await sel._select_codex_auto_agent_candidate(
            request=request,
            request_body={
                "model": review_alias,
                "previous_response_id": "resp-owned",
                "input": [{"type": "function_call", "name": "inspect"}],
            },
        )

    assert exc_info.value.status_code == 409
    detail = exc_info.value.detail
    assert detail["failure_phase"] == "session_owner_replay_unsafe_auto_review"
    owner_record_lookup.assert_not_awaited()
    assert sa.get_request_effective_session_identity(request) is None


@pytest.mark.asyncio
async def test_codex_owned_owner_cooldown_reselects_with_effective_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sel, candidate = _patch_codex_selector_basics(monkeypatch)
    selector_globals = sel._select_codex_auto_agent_candidate.__globals__
    request = _codex_selector_request("base-thread")
    owner_record = {
        "state": "owned",
        "owner": "owner-a",
        "attributes": {
            **candidate,
            "codex_oauth_account_label": "owner-account",
            "codex_oauth_account_hash": "owner-hash",
            "codex_oauth_lane_key": "owner-lane",
        },
    }

    async def _get_owner_record(
        *, session_identity: str | None, **_kwargs: Any
    ) -> tuple[Any, str, None]:
        if session_identity == "base-thread":
            return owner_record, "owner-key", None
        assert session_identity is not None
        assert session_identity.startswith("aawm-session-owner-redispatch-v1:")
        return None, "effective-key", None

    monkeypatch.setattr(sa, "get_session_owner_record", _get_owner_record)
    monkeypatch.setattr(
        sa,
        "owner_record_as_affinity_hint",
        lambda _record, **_kwargs: dict(owner_record["attributes"]),
    )
    monkeypatch.setitem(
        selector_globals,
        "_find_codex_auto_agent_affinity_candidate",
        lambda *_args, **_kwargs: dict(candidate),
    )
    monkeypatch.setitem(
        selector_globals,
        "_build_codex_auto_agent_affinity_candidate_state",
        AsyncMock(
            return_value={
                "candidate": dict(candidate),
                "cooldown_seconds": 10.0,
                "lane_key": "owner-lane",
            }
        ),
    )
    monkeypatch.setitem(
        selector_globals, "_candidate_matches_affinity", lambda *_args: False
    )
    activate = MagicMock(wraps=sa.activate_session_owner_redispatch_effective_identity)
    monkeypatch.setattr(sa, "activate_session_owner_redispatch_effective_identity", activate)

    selected = await sel._select_codex_auto_agent_candidate(
        request=request,
        request_body={
            "model": "alias",
            "litellm_metadata": {"redispatch_ordinal": 1},
            "input": [{"type": "function_call", "name": "inspect"}],
        },
    )

    assert selected["selection_reason"] == "first_available"
    assert selected["candidate"] == candidate
    assert selected["session_owner_identity"].startswith(
        "aawm-session-owner-redispatch-v1:"
    )
    assert sa.get_request_effective_session_identity(request) == selected[
        "session_owner_identity"
    ]
    activate.assert_called_once()


@pytest.mark.asyncio
async def test_codex_effective_identity_second_owned_conflict_stays_409(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sel, _candidate = _patch_codex_selector_basics(monkeypatch)
    selector_globals = sel._select_codex_auto_agent_candidate.__globals__
    request = _codex_selector_request("base-thread")
    effective_identity = sa.activate_session_owner_redispatch_effective_identity(
        request=request,
        base_session_identity="base-thread",
    )
    assert effective_identity is not None
    owner_record = {
        "state": "owned",
        "owner": "owner-b",
        "attributes": {"provider": "other", "model": "removed", "route_family": "other"},
    }
    monkeypatch.setattr(
        sa,
        "get_session_owner_record",
        AsyncMock(return_value=(owner_record, "effective-key", None)),
    )
    monkeypatch.setattr(
        sa,
        "owner_record_as_affinity_hint",
        lambda _record, **_kwargs: dict(owner_record["attributes"]),
    )
    monkeypatch.setitem(
        selector_globals,
        "_find_codex_auto_agent_affinity_candidate",
        lambda *_args, **_kwargs: None,
    )
    activate = MagicMock(wraps=sa.activate_session_owner_redispatch_effective_identity)
    monkeypatch.setattr(sa, "activate_session_owner_redispatch_effective_identity", activate)
    monkeypatch.setattr(
        sa,
        "raise_session_owner_redispatch_required",
        lambda **kwargs: (_ for _ in ()).throw(HTTPException(status_code=409, detail=kwargs)),
    )

    with pytest.raises(HTTPException) as exc_info:
        await sel._select_codex_auto_agent_candidate(
            request=request,
            request_body={"model": "alias", "input": [{"type": "reasoning"}]},
        )

    assert exc_info.value.status_code == 409
    activate.assert_not_called()


@pytest.mark.asyncio
async def test_auto_review_owners_are_distinct_and_free_selection_moves_candidates() -> None:
    attrs_a = _full_attrs(
        model="model-a",
        account_hash="acct-a",
        account_label="account-a",
        account_lane="lane-a",
    )
    attrs_b = _full_attrs(
        model="model-b",
        account_hash="acct-b",
        account_label="account-b",
        account_lane="lane-b",
    )
    parent_identity = "guardian-thread"
    owners: list[str] = []
    owner_keys: list[str] = []
    redis = _FakeRedisCache()

    for request_call, attributes in (
        ("review-call-one", attrs_a),
        ("review-call-two", attrs_b),
    ):
        request = type("Req", (), {})()
        request.state = type("State", (), {})()
        request.state.aawm_alias_request_litellm_call_id = request_call
        sa.activate_codex_auto_review_session_identity(
            request=request,
            parent_session_identity=parent_identity,
        )
        owner = sa.activate_codex_auto_review_session_owner_identity(
            request=request,
            parent_session_identity=parent_identity,
            request_call_identity=request_call,
        )
        assert owner is not None
        owners.append(owner)
        with _patch_dual(redis):
            guard = await sa.ensure_session_owner_guard_for_request(
                request=request,
                session_identity=owner,
                requested_attributes=attributes,
            )
            assert guard.cache_key is not None
            owner_keys.append(guard.cache_key)
            assert guard.decision is sa.SessionOwnerGuardDecision.UNOWNED_RESERVED
            assert guard.held_reservation is True
            lease = sa.get_request_session_owner_lease(request)
            assert lease is not None
            result = await sa.finalize_codex_auto_review_lease_on_success(lease)
            assert result is not None
            assert result.outcome is sa.SessionOwnerMutationOutcome.RELEASED
            assert lease.released is True

    assert owners[0] != owners[1]
    assert owner_keys[0] != owner_keys[1]


@pytest.mark.asyncio
async def test_concurrent_auto_review_owners_do_not_collide() -> None:
    redis = _FakeRedisCache()
    attributes = _full_attrs(
        account_hash="acct-concurrent",
        account_label="account-concurrent",
        account_lane="lane-concurrent",
    )

    async def _review(request_call: str) -> tuple[str, sa.SessionOwnerGuardResult]:
        request = type("Req", (), {})()
        request.state = type("State", (), {})()
        request.state.aawm_alias_request_litellm_call_id = request_call
        owner = sa.activate_codex_auto_review_session_owner_identity(
            request=request,
            parent_session_identity="guardian-thread",
            request_call_identity=request_call,
        )
        assert owner is not None
        with _patch_dual(redis):
            guard = await sa.guard_session_owner_before_egress(
                session_identity=owner,
                requested_attributes=attributes,
            )
            return owner, guard

    owners, guards = zip(*(await asyncio.gather(_review("same"), _review("other"))), strict=True)
    assert owners[0] != owners[1]
    assert [guard.decision for guard in guards] == [
        sa.SessionOwnerGuardDecision.UNOWNED_RESERVED,
        sa.SessionOwnerGuardDecision.UNOWNED_RESERVED,
    ]
    assert len(redis._data) == 2


@pytest.mark.asyncio
async def test_auto_review_lease_cleanup_does_not_accumulate_or_promote() -> None:
    redis = _FakeRedisCache()
    attributes = _full_attrs(
        account_hash="acct-cleanup",
        account_label="account-cleanup",
        account_lane="lane-cleanup",
    )
    request = type("Req", (), {})()
    request.state = type("State", (), {})()
    request.state.aawm_alias_request_litellm_call_id = "cleanup-call"
    owner = sa.activate_codex_auto_review_session_owner_identity(
        request=request,
        parent_session_identity="guardian-thread",
        request_call_identity="cleanup-call",
    )
    assert owner is not None

    with _patch_dual(redis):
        success_lease = None
        guard = await sa.ensure_session_owner_guard_for_request(
            request=request,
            session_identity=owner,
            requested_attributes=attributes,
        )
        assert guard.decision is sa.SessionOwnerGuardDecision.UNOWNED_RESERVED
        success_lease = sa.get_request_session_owner_lease(request)
        assert success_lease is not None
        success = await sa.finalize_codex_auto_review_lease_on_success(success_lease)
        assert success is not None
        assert success.outcome is sa.SessionOwnerMutationOutcome.RELEASED

        failure_guard = await sa.ensure_session_owner_guard_for_request(
            request=request,
            session_identity=owner,
            requested_attributes=attributes,
            raise_on_redispatch=False,
        )
        assert failure_guard.decision is sa.SessionOwnerGuardDecision.UNOWNED_RESERVED
        failure_lease = sa.get_request_session_owner_lease(request)
        assert failure_lease is not None
        failure = await sa.finalize_session_owner_lease_on_failure(failure_lease)
        assert failure is not None
        assert failure.outcome is sa.SessionOwnerMutationOutcome.RELEASED

        cancel_guard = await sa.ensure_session_owner_guard_for_request(
            request=request,
            session_identity=owner,
            requested_attributes=attributes,
            raise_on_redispatch=False,
        )
        assert cancel_guard.decision is sa.SessionOwnerGuardDecision.UNOWNED_RESERVED
        cancel_lease = sa.get_request_session_owner_lease(request)
        assert cancel_lease is not None
        cancellation = await sa.finalize_session_owner_lease_on_failure(cancel_lease)
        assert cancellation is not None
        assert cancellation.outcome is sa.SessionOwnerMutationOutcome.RELEASED

        snapshot = _redis_snapshot(redis)
        assert all(not data for data in snapshot)

    assert success_lease is not None
    assert failure_lease is not None
    assert cancel_lease is not None
    assert success_lease.promoted is False
    assert failure_lease.promoted is False
    assert cancel_lease.promoted is False


@pytest.mark.asyncio
async def test_codex_ordinary_ownership_promotes_durable_record() -> None:
    redis = _FakeRedisCache()
    attributes = _full_attrs()
    request = type("Req", (), {})()
    request.state = type("State", (), {})()

    with _patch_dual(redis):
        guard = await sa.ensure_session_owner_guard_for_request(
            request=request,
            session_identity="ordinary-session",
            requested_attributes=attributes,
        )
        assert guard.decision is sa.SessionOwnerGuardDecision.UNOWNED_RESERVED
        lease = sa.get_request_session_owner_lease(request)
        assert lease is not None
        result = await sa.finalize_session_owner_lease_on_success(
            lease,
            attributes=attributes,
        )
        assert result is not None
        assert result.outcome is sa.SessionOwnerMutationOutcome.PROMOTED

        owner_record, cache_key, error = await sa.get_session_owner_record(
            session_identity="ordinary-session",
            request=request,
        )
        assert error is None
        assert owner_record is not None
        assert owner_record["state"] == "owned"
        assert owner_record.get("reservation_token") is None
        assert cache_key == guard.cache_key
        assert set(redis._data) == {
            f"litellm:{cache_key}" if cache_key is not None else ""
        }

    sa.set_request_session_owner_lease(request, None)


@pytest.mark.asyncio
async def test_direct_openai_guard_retries_once_with_effective_identity_and_preserves_attributes() -> None:
    redis = _FakeRedisCache()
    request = _codex_selector_request("direct-base-thread")
    old_attributes = sa.build_session_owner_attributes(
        provider="openai",
        model="gpt-4o-mini",
        route_family="openai_responses",
        endpoint_contract="openai_responses",
        state_format="openai_responses",
        ingress="openai_passthrough",
        requested_model="gpt-4o-mini",
    )
    requested_attributes = sa.build_session_owner_attributes(
        provider="openai",
        model="gpt-5.4-codex",
        route_family="codex_oauth",
        endpoint_contract="openai_responses",
        state_format="openai_responses",
        ingress="openai_passthrough",
        requested_model="gpt-5.4-codex",
        extra={
            "account_label": "direct-account",
            "account_hash": "direct-hash",
            "account_lane": "direct-lane",
        },
    )
    body = {"model": "gpt-5.4-codex", "input": [{"type": "function_call"}]}

    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        initial = await sa.guard_session_owner_before_egress(
            session_identity="direct-base-thread",
            requested_attributes=old_attributes,
        )
        promoted = await sa.promote_session_owner_reservation(
            session_identity="direct-base-thread",
            reservation_token=initial.reservation_token,
            attributes=old_attributes,
        )
        assert promoted.outcome is sa.SessionOwnerMutationOutcome.PROMOTED

        original_guard = sa.guard_session_owner_before_egress
        with patch.object(
            sa,
            "guard_session_owner_before_egress",
            new=AsyncMock(side_effect=original_guard),
        ) as guard_mock:
            first = await sa.ensure_session_owner_guard_for_request(
                request=request,
                request_body=body,
                session_identity="direct-base-thread",
                requested_attributes=requested_attributes,
                require_exact_attributes=True,
                raise_on_redispatch=False,
            )
            assert sa.is_exact_owned_session_owner_route_mismatch(
                guard=first,
                requested_attributes=requested_attributes,
            )
            assert sa.clear_non_held_request_session_owner_lease(request) is True
            effective_identity = sa.activate_session_owner_redispatch_effective_identity(
                request=request,
                base_session_identity="direct-base-thread",
            )
            assert effective_identity is not None
            second = await sa.ensure_session_owner_guard_for_request(
                request=request,
                request_body=body,
                session_identity=effective_identity,
                requested_attributes=requested_attributes,
                require_exact_attributes=True,
                raise_on_redispatch=False,
            )

    assert first.decision is sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED
    assert second.held_reservation is True
    assert guard_mock.await_count == 2
    assert guard_mock.await_args_list[0].kwargs["requested_attributes"] == requested_attributes
    assert guard_mock.await_args_list[1].kwargs["requested_attributes"] == requested_attributes
    lease = sa.get_request_session_owner_lease(request)
    assert lease is not None
    assert lease.attributes == requested_attributes


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


@pytest.mark.asyncio
async def test_lease_renewal_survives_ttl_until_promotion() -> None:
    redis = _FakeRedisCache()
    attrs = _full_attrs()
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        guard = await sa.guard_session_owner_before_egress(
            session_identity="sess-renewal-long-operation",
            requested_attributes=attrs,
            reservation_ttl_seconds=30,
        )
        lease = sa.lease_from_guard_result(guard, attributes=attrs)
        real_sleep = asyncio.sleep

        async def advance_time_and_yield(delay: float) -> None:
            redis.advance_time(delay)
            await real_sleep(0)

        async def provider_operation() -> str:
            while redis.renewal_calls < 4:
                await real_sleep(0)
            return "provider response"

        with patch.object(
            sa.asyncio,
            "sleep",
            new=advance_time_and_yield,
        ):
            result = await sa.run_with_session_owner_lease_renewal(
                lease,
                provider_operation,
                reservation_ttl_seconds=30,
            )

        assert result == "provider response"
        assert redis.renewal_calls >= 4
        assert redis._clock > 30
        assert lease.renewal_task is None

        promoted = await sa.finalize_session_owner_lease_on_success(
            lease,
            attributes=attrs,
        )

    assert promoted is not None
    assert promoted.outcome is sa.SessionOwnerMutationOutcome.PROMOTED
    assert lease.promoted is True


@pytest.mark.asyncio
async def test_lease_renewal_finalizer_barrier_promotes_from_provider_operation() -> None:
    redis = _FakeRedisCache()
    attrs = _full_attrs()
    session_identity = "sess-renewal-finalizer-success"
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        guard = await sa.guard_session_owner_before_egress(
            session_identity=session_identity,
            requested_attributes=attrs,
        )
        lease = sa.lease_from_guard_result(guard, attributes=attrs)
        real_sleep = asyncio.sleep

        async def provider_operation() -> str:
            await real_sleep(0)
            assert lease.renewal_task is not None
            assert not lease.renewal_task.done()
            promoted = await sa.finalize_session_owner_lease_on_success(
                lease,
                attributes=attrs,
            )
            assert promoted is not None
            assert promoted.outcome is sa.SessionOwnerMutationOutcome.PROMOTED
            return "provider response"

        result = await sa.run_with_session_owner_lease_renewal(
            lease,
            provider_operation,
            reservation_ttl_seconds=30,
            renewal_interval_seconds=1,
        )
        record, _, error = await sa.get_session_owner_record(
            session_identity=session_identity,
        )

    assert result == "provider response"
    assert error is None
    assert record is not None
    assert record["state"] == sa.SessionOwnerRecordState.OWNED.value
    assert lease.promoted is True
    assert lease.released is False
    assert lease.renewal_task is None


@pytest.mark.asyncio
async def test_lease_renewal_finalizer_barrier_releases_from_provider_operation() -> None:
    redis = _FakeRedisCache()
    attrs = _full_attrs()
    session_identity = "sess-renewal-finalizer-failure"
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        guard = await sa.guard_session_owner_before_egress(
            session_identity=session_identity,
            requested_attributes=attrs,
        )
        lease = sa.lease_from_guard_result(guard, attributes=attrs)
        real_sleep = asyncio.sleep

        async def provider_operation() -> None:
            await real_sleep(0)
            assert lease.renewal_task is not None
            assert not lease.renewal_task.done()
            released = await sa.finalize_session_owner_lease_on_failure(lease)
            assert released is not None
            assert released.outcome is sa.SessionOwnerMutationOutcome.RELEASED
            raise RuntimeError("provider failed")

        with pytest.raises(RuntimeError, match="provider failed"):
            await sa.run_with_session_owner_lease_renewal(
                lease,
                provider_operation,
                reservation_ttl_seconds=30,
                renewal_interval_seconds=1,
            )
        record, _, error = await sa.get_session_owner_record(
            session_identity=session_identity,
        )

    assert error is None
    assert record is None
    assert lease.promoted is False
    assert lease.released is True
    assert lease.renewal_task is None


@pytest.mark.asyncio
async def test_lease_renewal_loss_cancels_operation_and_prevents_promotion() -> None:
    redis = _FakeRedisCache()
    attrs = _full_attrs()
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        guard = await sa.guard_session_owner_before_egress(
            session_identity="sess-renewal-lost",
            requested_attributes=attrs,
            reservation_ttl_seconds=30,
        )
        lease = sa.lease_from_guard_result(guard, attributes=attrs)
        operation_cancelled = asyncio.Event()
        real_sleep = asyncio.sleep
        never = asyncio.Event()

        async def provider_operation() -> None:
            try:
                await never.wait()
            except asyncio.CancelledError:
                operation_cancelled.set()
                raise

        async def fail_after_yield(_: float) -> None:
            redis.drop_reservation_on_renewal = True
            await real_sleep(0)

        with patch.object(sa.asyncio, "sleep", new=fail_after_yield):
            with pytest.raises(sa.SessionOwnerLeaseRenewalError):
                await sa.run_with_session_owner_lease_renewal(
                    lease,
                    provider_operation,
                    reservation_ttl_seconds=30,
                    renewal_interval_seconds=1,
                )

        assert operation_cancelled.is_set()
        assert redis.renewal_calls == 1
        assert lease.renewal_task is None

        promoted = await sa.promote_session_owner_reservation(
            session_identity=lease.session_identity,
            reservation_token=lease.reservation_token,
            attributes=attrs,
        )
        released = await sa.finalize_session_owner_lease_on_failure(lease)

    assert promoted.outcome is sa.SessionOwnerMutationOutcome.NOT_HELD
    assert released is not None
    assert released.outcome is sa.SessionOwnerMutationOutcome.NOT_HELD
    assert lease.released is True


@pytest.mark.asyncio
async def test_lease_renewal_cleans_up_after_operation_exception() -> None:
    redis = _FakeRedisCache()
    attrs = _full_attrs()
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        guard = await sa.guard_session_owner_before_egress(
            session_identity="sess-renewal-exception",
            requested_attributes=attrs,
        )
        lease = sa.lease_from_guard_result(guard, attributes=attrs)
        real_sleep = asyncio.sleep

        async def provider_operation() -> None:
            await real_sleep(0)
            raise RuntimeError("provider failed")

        with pytest.raises(RuntimeError, match="provider failed"):
            await sa.run_with_session_owner_lease_renewal(
                lease,
                provider_operation,
            )

        assert lease.renewal_task is None
        released = await sa.finalize_session_owner_lease_on_failure(lease)

    assert released is not None
    assert released.outcome is sa.SessionOwnerMutationOutcome.RELEASED
    assert lease.released is True


@pytest.mark.asyncio
async def test_lease_renewal_cleans_up_on_caller_cancellation() -> None:
    redis = _FakeRedisCache()
    attrs = _full_attrs()
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        guard = await sa.guard_session_owner_before_egress(
            session_identity="sess-renewal-cancelled",
            requested_attributes=attrs,
        )
        lease = sa.lease_from_guard_result(guard, attributes=attrs)
        operation_started = asyncio.Event()
        never = asyncio.Event()

        async def provider_operation() -> None:
            operation_started.set()
            await never.wait()

        task = asyncio.create_task(
            sa.run_with_session_owner_lease_renewal(
                lease,
                provider_operation,
            )
        )
        await operation_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert lease.renewal_task is None
        released = await sa.finalize_session_owner_lease_on_failure(lease)

    assert released is not None
    assert released.outcome is sa.SessionOwnerMutationOutcome.RELEASED
    assert lease.released is True


# ---------------------------------------------------------------------------
# Reopened D1-614: one proxy ERROR and one failure rollup for handled owner 409s.
# ---------------------------------------------------------------------------

_D614_SESSION_ID = "sess-d614-raw-1234567890"
_D614_OWNER_ID = "owner-raw-uuid-abcdef"
_D614_CACHE_KEY = "session_owner:d614-raw-cache-key"
_D614_RESERVATION_TOKEN = "raw-reservation-token-9999"
_D614_CALL_ID = "call-raw-id-777"
_D614_TRACE_ID = "trace-raw-id-888"
_D614_AGENT_ID = "agent-raw-id-666"
_D614_ACCOUNT_HASH = "chatgpt-account-hash:0123456789ab"
_D614_ACCOUNT_LANE = "chatgpt-account:0123456789ab"
_D614_RAW_CREDENTIAL = "sk-live-abcdef123456"
_D614_RAW_VALUES = (
    _D614_SESSION_ID,
    _D614_OWNER_ID,
    _D614_CACHE_KEY,
    _D614_RESERVATION_TOKEN,
    _D614_CALL_ID,
    _D614_TRACE_ID,
    _D614_AGENT_ID,
    _D614_ACCOUNT_HASH,
    _D614_ACCOUNT_LANE,
    _D614_RAW_CREDENTIAL,
)


def _d614_guard_result() -> "sa.SessionOwnerGuardResult":
    record = {
        "state": "owned",
        "owner": _D614_OWNER_ID,
        "attributes": {
            "provider": "openai",
            "model": "gpt-5.6-sol",
            "route_family": "codex_oauth",
            "endpoint_contract": "openai_responses",
            "state_format": "openai_responses",
            "account_hash": _D614_ACCOUNT_HASH,
            "account_lane": _D614_ACCOUNT_LANE,
            "account_label": "primary",
        },
        "reservation_token": _D614_RESERVATION_TOKEN,
    }
    return sa.SessionOwnerGuardResult(
        decision=sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
        session_identity=_D614_SESSION_ID,
        cache_key=_D614_CACHE_KEY,
        reservation_token=_D614_RESERVATION_TOKEN,
        owner_id=_D614_OWNER_ID,
        owner_record=record,
        mismatch_reason=(
            "session_owner mismatch owner=openai requested=anthropic "
            f"api_key={_D614_RAW_CREDENTIAL}"
        ),
        held_reservation=False,
    )


def test_d614_owner_409_emits_one_proxy_error_and_one_failure_rollup() -> None:  # noqa: PLR0915
    from types import SimpleNamespace

    state = SimpleNamespace(
        aawm_alias_request_context={
            "litellm_call_id": _D614_CALL_ID,
            "trace_id": _D614_TRACE_ID,
            "client_product_label": "codex-cli",
            "rollup_group_header_label": "repo codex-cli",
            "agent_dispatch": {"agent_id": _D614_AGENT_ID},
        },
        aawm_alias_request_litellm_call_id=_D614_CALL_ID,
    )
    request = SimpleNamespace(
        state=state,
        url=SimpleNamespace(path="/v1/responses"),
    )
    dedicated_warning = MagicMock()
    rollup = MagicMock(return_value=True)

    with patch.object(
        sa.verbose_proxy_logger, "error"
    ) as proxy_error, patch.object(
        sa.verbose_proxy_logger, "warning"
    ) as proxy_warning, patch(
        "litellm.proxy.aawm_route_logging.record_aawm_route_rollup_failure",
        rollup,
    ), patch.object(
        logging.getLogger("litellm.proxy.session_owner"),
        "warning",
        dedicated_warning,
    ), pytest.raises(HTTPException) as exc_info:
        sa.raise_session_owner_redispatch_required(
            session_identity=_D614_SESSION_ID,
            guard=_d614_guard_result(),
            alias_model="Codex-auto-agent",
            candidate={
                "provider": "anthropic",
                "model": "claude-opus-4-6",
                "route_family": "anthropic_native",
                "endpoint_contract": "anthropic_messages",
            },
            failure_phase="session_owner_mismatch",
            request=request,
        )

    proxy_error.assert_called_once()
    proxy_warning.assert_not_called()
    rollup.assert_called_once()
    dedicated_warning.assert_not_called()

    error_call = proxy_error.call_args
    assert error_call.args[0] == "AAWM_TERMINAL_ERROR: %s"
    error_fields = error_call.kwargs["extra"]
    assert error_fields["event_type"] == "redispatch_required"
    assert error_fields["status_code"] == 409
    assert error_fields["failure_class"] == "session_owner_redispatch"
    assert error_fields["failure_phase"] == "session_owner_mismatch"
    assert error_fields["attempted_provider_call"] is False
    assert error_fields["redispatch_required"] is True
    assert error_fields["correlation_id"].startswith("sha256:")

    summary = rollup.call_args.kwargs["message"]
    assert rollup.call_args.kwargs == {
        "message": summary,
        "status": "Failed",
    }
    assert "LiteLLM Proxy" in summary
    assert "HTTP 409" in summary
    assert "session-owner mismatch" in summary
    assert "redispatch" in summary
    assert "redispatch_required=true" in summary
    assert "attempted_provider_call=false" in summary
    assert "[REDACTED]" in summary
    assert len(summary) <= 480
    assert error_call.kwargs["exc_info"] is False

    rollup_kwargs = rollup.call_args.args[0]
    rollup_context = rollup_kwargs["litellm_params"]["metadata"][
        "aawm_route_rollup_context"
    ]
    assert rollup_context == {
        "group_header_label": "repo codex-cli",
        "incoming_endpoint": "/v1/responses",
        "outgoing_target": "anthropic_messages",
        "model_label": "Codex-auto-agent",
        "reasoning_effort": "none",
    }
    emitted = json.dumps(
        {
            "error": error_call.args,
            "error_kwargs": error_call.kwargs,
            "rollup": rollup.call_args,
        },
        default=str,
    )
    for raw in _D614_RAW_VALUES:
        assert raw not in emitted

    exc = exc_info.value
    assert exc.status_code == 409
    detail = exc.detail
    assert detail["redispatch_required"] is True
    assert detail["attempted_provider_call"] is False
    assert detail["failure_phase"] == "session_owner_mismatch"
    assert detail["error"]["code"] == "aawm_session_owner_redispatch_required"
    assert detail["error"]["type"] == "invalid_request_error"
    assert detail["canonical_session_identity"] == _D614_SESSION_ID
    assert detail["alias_model"] == "Codex-auto-agent"
    assert detail["redispatch_model"] == "Codex-auto-agent"
    assert detail["selected_provider"] == "openai"
    assert detail["selected_model"] == "gpt-5.6-sol"
    assert detail["selected_route_family"] == "codex_oauth"
    assert detail["session_owner"]["session_owner_state"] == "owned"
    assert _D614_RESERVATION_TOKEN not in json.dumps(detail)


def test_d614_nested_owner_409_uses_one_terminal_error_marker() -> None:
    from types import SimpleNamespace

    state = SimpleNamespace(
        aawm_alias_request_context={
            "litellm_call_id": _D614_CALL_ID,
            "trace_id": _D614_TRACE_ID,
        },
        aawm_alias_request_litellm_call_id=_D614_CALL_ID,
    )
    request = SimpleNamespace(
        state=state,
        url=SimpleNamespace(path="/v1/responses"),
    )
    rollup = MagicMock(return_value=True)

    with patch.object(sa.verbose_proxy_logger, "error") as proxy_error, patch(
        "litellm.proxy.aawm_route_logging.record_aawm_route_rollup_failure",
        rollup,
    ):
        for _ in range(2):
            with pytest.raises(HTTPException) as exc_info:
                sa.raise_session_owner_redispatch_required(
                    session_identity=_D614_SESSION_ID,
                    guard=_d614_guard_result(),
                    alias_model="Codex-auto-agent",
                    failure_phase="session_owner_mismatch",
                    request=request,
                )
            assert exc_info.value.status_code == 409

    proxy_error.assert_called_once()
    assert rollup.call_count == 2


def _d614_raise_with_observability(
    *,
    error_side_effect: Optional[BaseException] = None,
    rollup_side_effect: Optional[BaseException] = None,
) -> tuple[HTTPException, MagicMock, MagicMock]:
    rollup = MagicMock(side_effect=rollup_side_effect, return_value=True)
    with patch.object(
        sa.verbose_proxy_logger, "error", side_effect=error_side_effect
    ) as proxy_error, patch(
        "litellm.proxy.aawm_route_logging.record_aawm_route_rollup_failure",
        rollup,
    ), pytest.raises(HTTPException) as exc_info:
        sa.raise_session_owner_redispatch_required(
            session_identity=_D614_SESSION_ID,
            guard=_d614_guard_result(),
            alias_model="Codex-auto-agent",
            failure_phase="session_owner_mismatch",
        )
    return exc_info.value, proxy_error, rollup


def test_d614_proxy_error_failure_still_raises_identical_409() -> None:
    baseline, _, _ = _d614_raise_with_observability()
    failed, proxy_warning, rollup = _d614_raise_with_observability(
        error_side_effect=RuntimeError("logging backend down"),
    )
    proxy_warning.assert_called_once()
    rollup.assert_called_once()
    assert failed.status_code == baseline.status_code == 409
    assert failed.detail == baseline.detail


def test_d614_rollup_failure_still_raises_identical_409() -> None:
    baseline, _, _ = _d614_raise_with_observability()
    failed, proxy_error, rollup = _d614_raise_with_observability(
        rollup_side_effect=RuntimeError("rollup backend down"),
    )
    proxy_error.assert_called_once()
    rollup.assert_called_once()
    assert failed.status_code == baseline.status_code == 409
    assert failed.detail == baseline.detail


def _d614_uvicorn_access_record(
    *,
    client: str,
    method: str,
    path: str,
    status: int,
    http_version: str = "1.1",
) -> logging.LogRecord:
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=(client, method, path, http_version, status),
        exc_info=None,
    )


def test_d614_owner_409_consumes_leftover_uvicorn_access() -> None:
    """Direct/nested session-owner 409 still consumes leftover uvicorn ACCESS.

    Live 2026-08-24: concurrent reservation raised HTTP 409 from the
    OpenAI passthrough handler before ``pass_through_request`` registered
    ACCESS replacement, so uvicorn printed
    ``INFO: … "POST /openai_passthrough/responses HTTP/1.1" 409 Conflict``.
    Distinct leftover source ports were separate requests; each exact POST
    responses 409 registers once. A second identical ACCESS record stays
    visible.
    """

    from fastapi import Request
    from starlette.datastructures import Headers

    from litellm._logging import (
        AawmRouteAccessLogReplacementFilter,
        clear_aawm_route_access_log_replacements,
    )
    from litellm.proxy.aawm_route_logging import clear_aawm_route_rollups

    clear_aawm_route_access_log_replacements()
    clear_aawm_route_rollups()
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.url = "http://127.0.0.1:4011/openai_passthrough/responses"
    request.headers = Headers({"user-agent": "codex-cli/0.149.1"})
    request.scope = {
        "type": "http",
        "method": "POST",
        "path": "/openai_passthrough/responses",
        "query_string": b"",
        "client": ("172.18.0.1", 35106),
        "http_version": "1.1",
    }
    with pytest.raises(HTTPException) as exc_info:
        sa.raise_session_owner_redispatch_required(
            session_identity=_D614_SESSION_ID,
            guard=_d614_guard_result(),
            alias_model="gpt-5.6-luna",
            failure_phase="session_owner_mismatch",
            request=request,
        )
    assert exc_info.value.status_code == 409
    leftover_409 = _d614_uvicorn_access_record(
        client="172.18.0.1:35106",
        method="POST",
        path="/openai_passthrough/responses",
        status=409,
    )
    access_filter = AawmRouteAccessLogReplacementFilter()
    assert access_filter.filter(leftover_409) is False
    assert access_filter.filter(leftover_409) is True
    clear_aawm_route_access_log_replacements()
    clear_aawm_route_rollups()


def test_d614_owner_409_on_non_post_responses_keeps_leftover_uvicorn_access() -> None:
    """Non-POST /openai_passthrough/responses 409 must not consume ACCESS."""

    from fastapi import Request
    from starlette.datastructures import Headers

    from litellm._logging import (
        AawmRouteAccessLogReplacementFilter,
        clear_aawm_route_access_log_replacements,
    )
    from litellm.proxy.aawm_route_logging import clear_aawm_route_rollups

    clear_aawm_route_access_log_replacements()
    clear_aawm_route_rollups()
    request = MagicMock(spec=Request)
    request.method = "GET"
    request.url = "http://127.0.0.1:4011/openai_passthrough/responses"
    request.headers = Headers({"user-agent": "codex-cli/0.149.1"})
    request.scope = {
        "type": "http",
        "method": "GET",
        "path": "/openai_passthrough/responses",
        "query_string": b"",
        "client": ("172.18.0.1", 35109),
        "http_version": "1.1",
    }
    with pytest.raises(HTTPException) as exc_info:
        sa.raise_session_owner_redispatch_required(
            session_identity=_D614_SESSION_ID,
            guard=_d614_guard_result(),
            alias_model="gpt-5.6-luna",
            failure_phase="session_owner_mismatch",
            request=request,
        )
    assert exc_info.value.status_code == 409
    leftover = _d614_uvicorn_access_record(
        client="172.18.0.1:35109",
        method="GET",
        path="/openai_passthrough/responses",
        status=409,
    )
    access_filter = AawmRouteAccessLogReplacementFilter()
    assert access_filter.filter(leftover) is True
    clear_aawm_route_access_log_replacements()
    clear_aawm_route_rollups()


def test_d614_owner_409_missing_scope_method_keeps_leftover_uvicorn_access() -> None:
    """Missing/falsy scope method must not consume ACCESS even if request.method is POST."""

    from fastapi import Request
    from starlette.datastructures import Headers

    from litellm._logging import (
        AawmRouteAccessLogReplacementFilter,
        clear_aawm_route_access_log_replacements,
    )
    from litellm.proxy.aawm_route_logging import clear_aawm_route_rollups

    clear_aawm_route_access_log_replacements()
    clear_aawm_route_rollups()
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.url = "http://127.0.0.1:4011/openai_passthrough/responses"
    request.headers = Headers({"user-agent": "codex-cli/0.149.1"})
    request.scope = {
        "type": "http",
        "method": "",
        "path": "/openai_passthrough/responses",
        "query_string": b"",
        "client": ("172.18.0.1", 35110),
        "http_version": "1.1",
    }
    with pytest.raises(HTTPException) as exc_info:
        sa.raise_session_owner_redispatch_required(
            session_identity=_D614_SESSION_ID,
            guard=_d614_guard_result(),
            alias_model="gpt-5.6-luna",
            failure_phase="session_owner_mismatch",
            request=request,
        )
    assert exc_info.value.status_code == 409
    leftover = _d614_uvicorn_access_record(
        client="172.18.0.1:35110",
        method="POST",
        path="/openai_passthrough/responses",
        status=409,
    )
    access_filter = AawmRouteAccessLogReplacementFilter()
    assert access_filter.filter(leftover) is True
    clear_aawm_route_access_log_replacements()
    clear_aawm_route_rollups()


def test_d614_owner_409_on_non_responses_path_keeps_leftover_uvicorn_access() -> None:
    """Non-responses session-owner 409 must not consume leftover uvicorn ACCESS."""

    from fastapi import Request
    from starlette.datastructures import Headers

    from litellm._logging import (
        AawmRouteAccessLogReplacementFilter,
        clear_aawm_route_access_log_replacements,
    )
    from litellm.proxy.aawm_route_logging import clear_aawm_route_rollups

    clear_aawm_route_access_log_replacements()
    clear_aawm_route_rollups()
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.url = "http://127.0.0.1:4011/grok/v1/responses"
    request.headers = Headers({"user-agent": "grok-cli/0.1.0"})
    request.scope = {
        "type": "http",
        "method": "POST",
        "path": "/grok/v1/responses",
        "query_string": b"",
        "client": ("172.18.0.1", 35107),
        "http_version": "1.1",
    }
    with pytest.raises(HTTPException) as exc_info:
        sa.raise_session_owner_redispatch_required(
            session_identity=_D614_SESSION_ID,
            guard=_d614_guard_result(),
            alias_model="grok-4",
            failure_phase="session_owner_mismatch",
            request=request,
        )
    assert exc_info.value.status_code == 409
    leftover = _d614_uvicorn_access_record(
        client="172.18.0.1:35107",
        method="POST",
        path="/grok/v1/responses",
        status=409,
    )
    access_filter = AawmRouteAccessLogReplacementFilter()
    assert access_filter.filter(leftover) is True
    clear_aawm_route_access_log_replacements()
    clear_aawm_route_rollups()


def test_d614_owner_409_on_lookalike_responses_path_keeps_leftover_uvicorn_access() -> None:
    """Look-alike responses paths must not consume leftover uvicorn ACCESS."""

    from fastapi import Request
    from starlette.datastructures import Headers

    from litellm._logging import (
        AawmRouteAccessLogReplacementFilter,
        clear_aawm_route_access_log_replacements,
    )
    from litellm.proxy.aawm_route_logging import clear_aawm_route_rollups

    clear_aawm_route_access_log_replacements()
    clear_aawm_route_rollups()
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.url = "http://127.0.0.1:4011/openai_passthrough/responses2"
    request.headers = Headers({"user-agent": "codex-cli/0.149.1"})
    request.scope = {
        "type": "http",
        "method": "POST",
        "path": "/openai_passthrough/responses2",
        "query_string": b"",
        "client": ("172.18.0.1", 35108),
        "http_version": "1.1",
    }
    with pytest.raises(HTTPException) as exc_info:
        sa.raise_session_owner_redispatch_required(
            session_identity=_D614_SESSION_ID,
            guard=_d614_guard_result(),
            alias_model="gpt-5.6-luna",
            failure_phase="session_owner_mismatch",
            request=request,
        )
    assert exc_info.value.status_code == 409
    leftover = _d614_uvicorn_access_record(
        client="172.18.0.1:35108",
        method="POST",
        path="/openai_passthrough/responses2",
        status=409,
    )
    access_filter = AawmRouteAccessLogReplacementFilter()
    assert access_filter.filter(leftover) is True
    clear_aawm_route_access_log_replacements()
    clear_aawm_route_rollups()


@pytest.mark.asyncio
async def test_d614_full_wrapper_records_owner_409_once_before_egress() -> None:
    from types import SimpleNamespace

    from starlette.datastructures import Headers, QueryParams

    from litellm.proxy._types import ProxyException, UserAPIKeyAuth
    from litellm.proxy.pass_through_endpoints import pass_through_endpoints as pte

    redis = _FakeRedisCache()
    owner_attrs = _full_attrs(
        route_family="openai",
        account_hash=None,
        account_label=None,
        account_lane=None,
    )
    state = SimpleNamespace(
        aawm_alias_request_context={
            "litellm_call_id": _D614_CALL_ID,
            "trace_id": _D614_TRACE_ID,
            "rollup_group_header_label": "repo codex-cli",
            "agent_dispatch": {"agent_id": _D614_AGENT_ID},
        },
        aawm_alias_request_litellm_call_id=_D614_CALL_ID,
    )
    request = SimpleNamespace(
        state=state,
        method="POST",
        headers=Headers(
            {
                "authorization": f"Bearer {_D614_RAW_CREDENTIAL}",
                "content-type": "application/json",
            }
        ),
        query_params=QueryParams({}),
        url=httpx.URL("http://localhost/v1/messages"),
    )
    provider_send = AsyncMock()
    proxy_logging = MagicMock()
    proxy_logging.pre_call_hook = AsyncMock(side_effect=lambda **kw: kw["data"])
    proxy_logging.post_call_failure_hook = AsyncMock(return_value=None)
    rollup = MagicMock(return_value=True)
    dedicated_warning = MagicMock()
    session_id = "sess-d614-no-egress"

    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        initial = await sa.guard_session_owner_before_egress(
            session_identity=session_id,
            requested_attributes=owner_attrs,
        )
        await sa.promote_session_owner_reservation(
            session_identity=session_id,
            reservation_token=initial.reservation_token,
            attributes=owner_attrs,
        )

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client",
            return_value=MagicMock(client=MagicMock()),
        ), patch.object(
            pte.HttpPassThroughEndpointHelpers,
            "non_streaming_http_request_handler",
            new=provider_send,
        ), patch(
            "litellm.proxy.proxy_server.proxy_logging_obj",
            proxy_logging,
        ), patch.object(
            pte,
            "emit_aawm_route_access_log",
        ), patch.object(
            pte.ProxyBaseLLMRequestProcessing,
            "get_custom_headers",
            return_value={},
        ), patch.object(
            pte,
            "_direct_capture_xai_passthrough_failure",
            new=AsyncMock(),
        ), patch.object(
            sa.verbose_proxy_logger, "error"
        ) as proxy_error, patch.object(
            sa.verbose_proxy_logger, "warning"
        ) as proxy_warning, patch(
            "litellm.proxy.aawm_route_logging.record_aawm_route_rollup_failure",
            rollup,
        ), patch.object(
            pte,
            "record_aawm_route_rollup_failure",
            rollup,
        ), patch.object(
            logging.getLogger("litellm.proxy.session_owner"),
            "warning",
            dedicated_warning,
        ), pytest.raises(ProxyException) as exc_info:
            await pte.pass_through_request(
                request=request,
                target="https://api.anthropic.com/v1/messages",
                custom_headers={"authorization": f"Bearer {_D614_RAW_CREDENTIAL}"},
                user_api_key_dict=UserAPIKeyAuth(),
                custom_body={
                    "model": "claude-opus-4-6",
                    "litellm_metadata": {
                        "session_id": session_id,
                    },
                },
                custom_llm_provider="anthropic",
                egress_credential_family="anthropic",
                expected_target_family="anthropic",
                stream=False,
            )

    proxy_error.assert_called_once()
    proxy_warning.assert_not_called()
    rollup.assert_called_once()
    dedicated_warning.assert_not_called()
    provider_send.assert_not_awaited()
    error_call = proxy_error.call_args
    assert error_call.args[0] == "AAWM_TERMINAL_ERROR: %s"
    error_fields = error_call.kwargs["extra"]
    assert error_fields["event_type"] == "redispatch_required"
    assert error_fields["status_code"] == 409
    assert error_fields["attempted_provider_call"] is False
    assert error_fields["redispatch_required"] is True
    assert error_call.kwargs["exc_info"] is False
    summary = rollup.call_args.kwargs["message"]
    assert rollup.call_args.kwargs == {
        "message": summary,
        "status": "Failed",
    }
    rollup_context = rollup.call_args.args[0]["litellm_params"]["metadata"][
        "aawm_route_rollup_context"
    ]
    assert rollup_context["group_header_label"] == "repo codex-cli"

    emitted = json.dumps(
        {
            "error": error_call.args,
            "error_kwargs": error_call.kwargs,
            "rollup": rollup.call_args,
        },
        default=str,
    )
    for raw in _D614_RAW_VALUES:
        assert raw not in emitted
    assert session_id not in emitted

    exc = exc_info.value
    assert exc.code == "409"
    assert exc.detail["error"]["code"] == "aawm_session_owner_redispatch_required"
    assert exc.detail["redispatch_required"] is True
    assert exc.detail["attempted_provider_call"] is False
    assert exc.detail["canonical_session_identity"] == session_id
    assert proxy_logging.post_call_failure_hook.await_args.kwargs["traceback_str"] is None


def test_interchangeable_openai_owner_ignores_account_identity_only() -> None:
    account1 = sa.build_session_owner_attributes(
        provider="openai",
        model="gpt-5.6-sol",
        route_family="codex_responses",
        account_label="account1",
        account_hash="hash-account-1",
        account_lane="codex-oauth:account1:hash-account-1",
        endpoint_contract="codex_responses",
        state_format="codex_responses",
        credential_affinity="interchangeable",
    )
    account2 = sa.build_session_owner_attributes(
        provider="openai",
        model="gpt-5.6-sol",
        route_family="codex_responses",
        account_label="account2",
        account_hash="hash-account-2",
        account_lane="codex-oauth:account2:hash-account-2",
        endpoint_contract="codex_responses",
        state_format="codex_responses",
        credential_affinity="interchangeable",
    )

    assert sa.build_session_owner_id(attributes=account1) == (
        sa.build_session_owner_id(attributes=account2)
    )
    assert sa._attributes_exactly_equal(left=account1, right=account2)
    hint = sa.owner_record_as_affinity_hint(
        {
            "state": "owned",
            "owner": "owner",
            "attributes": account1,
        }
    )
    assert hint is not None
    assert "codex_oauth_account_label" not in hint

    assert sa._attributes_exactly_equal(
        left=account1,
        right=dict(account2, model="gpt-5.6-terra"),
    )


def test_bound_codex_oauth_affinity_matches_legacy_account2_owner() -> None:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        codex_oauth,
    )

    request = _codex_selector_request("legacy-account2-owner")
    candidate = {
        "codex_oauth_account_label": "account1",
        "codex_oauth_account_hash": "hash-account-1",
        "codex_oauth_lane_key": "codex-oauth:account1:hash-account-1",
        "model": "gpt-5.6-sol",
        "codex_oauth_credential_affinity": "interchangeable",
        "access_token": "secret-token",
    }
    bound = codex_oauth._bind_codex_oauth_candidate_to_request(
        request,
        candidate,
    )
    identity = codex_oauth._get_bound_codex_oauth_candidate_identity(request)
    assert bound is not None
    assert identity is not None
    assert bound["credential_affinity"] == "interchangeable"
    assert identity["credential_affinity"] == "interchangeable"
    assert "access_token" not in bound
    assert "access_token" not in identity

    owner_attrs = sa.build_session_owner_attributes(
        provider="openai",
        model="gpt-5.6-sol",
        route_family="codex_oauth",
        account_label="account2",
        account_hash="hash-account-2",
        account_lane="codex-oauth:account2:hash-account-2",
        endpoint_contract="openai_responses",
        state_format="openai_responses",
    )
    assert "credential_affinity" not in owner_attrs
    requested_attrs = sa.build_session_owner_attributes(
        provider="openai",
        model="gpt-5.6-sol",
        route_family="codex_oauth",
        endpoint_contract="openai_responses",
        state_format="openai_responses",
        extra={
            key: value
            for key, value in {
                "account_label": identity.get("account_label"),
                "account_hash": identity.get("account_hash"),
                "account_lane": identity.get("lane_key"),
                "credential_affinity": identity.get("credential_affinity"),
            }.items()
            if value
        },
    )
    assert requested_attrs.get("credential_affinity") == "interchangeable"
    assert sa._attributes_exactly_equal(left=owner_attrs, right=requested_attrs)
    assert (
        sa._compatibility_mismatch_reason(
            owner_record={
                "state": "owned",
                "owner": "owner",
                "attributes": owner_attrs,
            },
            requested_attributes=requested_attrs,
            require_exact_attributes=True,
        )
        is None
    )


def _managed_direct_openai_owner_attrs(
    *,
    route_family: str,
    endpoint_contract: str,
    state_format: str,
    model: str = "gpt-5.6-sol",
    account_hash: str = "acct-1",
    account_label: str = "primary",
    account_lane: str = "lane-a",
    provider: str = "openai",
) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "route_family": route_family,
        "endpoint_contract": endpoint_contract,
        "state_format": state_format,
        "account_hash": account_hash,
        "account_label": account_label,
        "account_lane": account_lane,
    }


def _legacy_managed_direct_openai_owner_attrs(**over: Any) -> dict[str, Any]:
    attrs = _managed_direct_openai_owner_attrs(
        route_family="codex_responses",
        endpoint_contract="codex_responses",
        state_format="codex_responses",
    )
    attrs.update(over)
    return attrs


def _current_managed_direct_openai_owner_attrs(**over: Any) -> dict[str, Any]:
    attrs = _managed_direct_openai_owner_attrs(
        route_family="codex_oauth",
        endpoint_contract="openai_responses",
        state_format="openai_responses",
    )
    attrs.update(over)
    return attrs


def _owned_record(attributes: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": "owned",
        "owner": "owner",
        "attributes": attributes,
    }


def test_managed_direct_openai_owner_shapes_are_equivalent() -> None:
    historical = _legacy_managed_direct_openai_owner_attrs()
    current = _current_managed_direct_openai_owner_attrs()
    public = _managed_direct_openai_owner_attrs(
        route_family="openai_responses",
        endpoint_contract="openai_responses",
        state_format="openai_responses",
    )

    assert sa._managed_direct_openai_owner_shapes_are_equivalent(historical, current)
    assert sa._managed_direct_openai_owner_shapes_are_equivalent(current, historical)
    assert not sa._managed_direct_openai_owner_shapes_are_equivalent(historical, public)
    assert not sa._managed_direct_openai_owner_shapes_are_equivalent(current, public)
    assert sa._attributes_exactly_equal(left=historical, right=current)
    assert sa._attributes_exactly_equal(left=current, right=historical)
    assert not sa._attributes_exactly_equal(left=historical, right=public)
    assert sa._compatibility_mismatch_reason(
        owner_record=_owned_record(historical),
        requested_attributes=current,
        require_exact_attributes=True,
    ) is None
    assert sa._compatibility_mismatch_reason(
        owner_record=_owned_record(current),
        requested_attributes=historical,
        require_exact_attributes=False,
    ) is None
    assert sa.build_session_owner_id(attributes=historical) == (
        sa.build_session_owner_id(attributes=current)
    )
    assert sa.owner_record_as_affinity_hint(_owned_record(historical))["route_family"] == (
        "codex_responses"
    )
    assert sa.owner_record_as_affinity_hint(_owned_record(current))["route_family"] == (
        "codex_oauth"
    )


def test_managed_direct_openai_owner_shapes_remain_strict_nearby() -> None:
    owner = _legacy_managed_direct_openai_owner_attrs()
    compatible = [
        _current_managed_direct_openai_owner_attrs(model="gpt-5.6-terra"),
        _current_managed_direct_openai_owner_attrs(account_lane="lane-b"),
    ]
    incompatible = [
        _current_managed_direct_openai_owner_attrs(provider="xai"),
        _managed_direct_openai_owner_attrs(
            route_family="openai_responses",
            endpoint_contract="openai_responses",
            state_format="openai_responses",
        ),
        _managed_direct_openai_owner_attrs(
            route_family="codex_oauth",
            endpoint_contract="codex_responses",
            state_format="codex_responses",
        ),
        _managed_direct_openai_owner_attrs(
            route_family="codex_responses",
            endpoint_contract="openai_responses",
            state_format="openai_responses",
        ),
        _managed_direct_openai_owner_attrs(
            route_family="codex_oauth",
            endpoint_contract="openai_responses",
            state_format="codex_responses",
        ),
    ]
    for requested in compatible:
        assert sa._attributes_exactly_equal(left=owner, right=requested)
        assert sa._compatibility_mismatch_reason(
            owner_record=_owned_record(owner),
            requested_attributes=requested,
            require_exact_attributes=True,
        ) is None
        assert sa._compatibility_mismatch_reason(
            owner_record=_owned_record(owner),
            requested_attributes=requested,
            require_exact_attributes=False,
        ) is None
    for requested in incompatible:
        assert not sa._attributes_exactly_equal(left=owner, right=requested)
        assert sa._compatibility_mismatch_reason(
            owner_record=_owned_record(owner),
            requested_attributes=requested,
            require_exact_attributes=True,
        ) is not None
        assert sa._compatibility_mismatch_reason(
            owner_record=_owned_record(owner),
            requested_attributes=requested,
            require_exact_attributes=False,
        ) is not None


@pytest.mark.asyncio
async def test_direct_openai_owner_accepts_legacy_and_current_managed_shapes() -> None:
    redis = _FakeRedisCache()
    historical = _legacy_managed_direct_openai_owner_attrs()
    current = _current_managed_direct_openai_owner_attrs()
    public = _managed_direct_openai_owner_attrs(
        route_family="openai_responses",
        endpoint_contract="openai_responses",
        state_format="openai_responses",
    )
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ):
        reserved = await sa.guard_session_owner_before_egress(
            session_identity="sess-openai-015-compat",
            requested_attributes=historical,
        )
        promoted = await sa.promote_session_owner_reservation(
            session_identity="sess-openai-015-compat",
            reservation_token=reserved.reservation_token,
            attributes=historical,
        )
        assert promoted.outcome is sa.SessionOwnerMutationOutcome.PROMOTED
        continuation = await sa.ensure_session_owner_guard_for_request(
            session_identity="sess-openai-015-compat",
            requested_attributes=current,
            require_exact_attributes=True,
        )
        public_guard = await sa.ensure_session_owner_guard_for_request(
            session_identity="sess-openai-015-compat",
            requested_attributes=public,
            require_exact_attributes=True,
            raise_on_redispatch=False,
        )
        stored = await sa.get_session_owner_record(
            session_identity="sess-openai-015-compat"
        )

    assert continuation.decision is sa.SessionOwnerGuardDecision.COMPATIBLE_OWNER
    assert public_guard.decision is sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED
    assert "exactly match" in (public_guard.mismatch_reason or "")
    assert stored[0] is not None
    assert stored[0]["attributes"]["route_family"] == "codex_responses"
    assert stored[0]["attributes"]["endpoint_contract"] == "codex_responses"
    assert stored[0]["attributes"]["state_format"] == "codex_responses"


# =============================================================================
# CURSOR-015: Validated cursor replay state helpers
# =============================================================================

class TestCursor015ValidatedCursorReplay:
    def test_has_validated_cursor_replay_returns_false_for_fresh_request(self) -> None:
        from types import SimpleNamespace
        request = SimpleNamespace(state=SimpleNamespace())
        assert sa.has_validated_cursor_replay(request) is False

    def test_set_and_has_validated_cursor_replay(self) -> None:
        from types import SimpleNamespace
        request = SimpleNamespace(state=SimpleNamespace())
        body = {"model": "work", "input": []}
        sa.set_validated_cursor_replay(
            request,
            body=body,
            stage="cursor_replay_built",
            reason="strict_reconstruction_validated",
        )
        assert sa.has_validated_cursor_replay(request) is True
        assert sa.get_validated_cursor_replay_body_id(request) == id(body)

    def test_validated_cursor_replay_body_id_is_none_for_fresh_request(self) -> None:
        from types import SimpleNamespace
        request = SimpleNamespace(state=SimpleNamespace())
        assert sa.get_validated_cursor_replay_body_id(request) is None

    def test_has_validated_cursor_replay_returns_false_for_none_request(self) -> None:
        assert sa.has_validated_cursor_replay(None) is False

    def test_set_validated_cursor_replay_is_noop_for_none_request(self) -> None:
        sa.set_validated_cursor_replay(
            None,
            body={},
            stage="test",
            reason="test",
        )


# =============================================================================
# CURSOR-015: Derive effective identity
# =============================================================================

class TestCursor015DeriveEffectiveIdentity:
    def test_derives_deterministic_identity_for_base(self) -> None:
        identity = sa._derive_effective_identity_for_base("sess-cursor015-base")
        assert identity is not None
        assert identity.startswith(
            sa._SESSION_OWNER_REDISPATCH_EFFECTIVE_IDENTITY_PREFIX
        )
        assert identity == sa._derive_effective_identity_for_base("sess-cursor015-base")

    def test_returns_none_for_empty_base(self) -> None:
        assert sa._derive_effective_identity_for_base(None) is None
        assert sa._derive_effective_identity_for_base("") is None

    def test_different_bases_produce_different_identities(self) -> None:
        id_a = sa._derive_effective_identity_for_base("sess-a")
        id_b = sa._derive_effective_identity_for_base("sess-b")
        assert id_a is not None
        assert id_b is not None
        assert id_a != id_b


# =============================================================================
# CURSOR-015: Rediscovered effective owner
# =============================================================================

class TestCursor015RediscoveredEffectiveOwner:
    def test_set_rediscovered_effective_owner_stores_on_request(self) -> None:
        from types import SimpleNamespace
        request = SimpleNamespace(state=SimpleNamespace())
        sa.set_rediscovered_effective_owner(
            request,
            effective_identity="sess-cursor015-effective",
            owner_record={"owner": "owner-id", "state": "owned"},
        )
        stored = getattr(
            request.state,
            sa._REQUEST_STATE_REDISCOVERED_EFFECTIVE_OWNER_ATTR,
            None,
        )
        assert stored is not None
        assert stored["effective_identity"] == "sess-cursor015-effective"
        assert stored["owner_id"] == "owner-id"
        assert stored["state"] == "owned"

    def test_set_rediscovered_effective_owner_is_noop_for_none_request(self) -> None:
        sa.set_rediscovered_effective_owner(
            None,
            effective_identity="sess-cursor015-effective",
            owner_record={"owner": "x", "state": "owned"},
        )
