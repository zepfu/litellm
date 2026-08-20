"""OPENAI-020: same hosted-provider model and OpenAI account are mutable."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import pytest
from starlette.datastructures import State

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    session_affinity as sa,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import durable as durable_mod


REPO_ROOT = Path(__file__).resolve().parents[4]
HANDLER_PATH = (
    REPO_ROOT
    / "litellm/proxy/pass_through_endpoints/aawm_adapter_runtime/openai_passthrough_handler.py"
)
SID = "sess-openai020-canonical"


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
        key = args[0]
        raw = self._parent._data.get(key)
        if "PERSIST" in script and "reservation_token" in script and "owned" in script:
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
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}
        self.set_calls: list[Any] = []
        self.namespace = "litellm"
        self._data: dict[str, bytes] = {}
        self._ttl: dict[str, float] = {}
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

    async def async_get_cache(self, key, **kwargs):
        namespaced = self.check_and_fix_namespace(key)
        raw = self._data.get(namespaced)
        parsed = self._get_cache_logic(raw)
        if parsed is not None:
            self.store[key] = parsed
        return parsed if parsed is not None else self.store.get(key)

    async def async_set_cache(self, key, value, **kwargs):
        self.set_calls.append((key, value, kwargs))
        namespaced = self.check_and_fix_namespace(key)
        nx = bool(kwargs.get("nx"))
        ttl = kwargs.get("ttl") or kwargs.get("ex")
        claimed = await self._client.set(
            name=namespaced,
            value=json.dumps(value),
            nx=nx,
            ex=ttl,
        )
        if claimed:
            self.store[key] = value
        return claimed

    async def async_delete_cache(self, key, **kwargs):
        self.store.pop(key, None)
        namespaced = self.check_and_fix_namespace(key)
        await self._client.delete(namespaced)


class _FakeDualCache:
    def __init__(self, redis_cache: Optional[_FakeRedisCache]) -> None:
        self.redis_cache = redis_cache


def _patch_dual(redis):
    dual = None if redis is None else _FakeDualCache(redis)
    return patch.object(
        durable_mod, "get_aawm_alias_routing_dual_cache", return_value=dual
    )


def _request(session_id: str = SID):
    req = type("Req", (), {})()
    req.state = State()
    req.headers = {"session_id": session_id}
    return req


def _openai_attrs(*, model: str, account: str, **overrides: Any) -> dict[str, Any]:
    attrs = sa.build_session_owner_attributes(
        provider="openai",
        model=model,
        route_family="codex_responses",
        account_label=account,
        account_hash=f"hash-{account}",
        account_lane=f"codex-oauth:{account}:hash-{account}",
        endpoint_contract="codex_responses",
        state_format="codex_responses",
    )
    if overrides:
        return {**attrs, **overrides}
    return attrs


def _provider_attrs(
    *,
    provider: str,
    model: str,
    account: str,
    route_family: str = "codex_responses",
    endpoint_contract: str = "codex_responses",
    state_format: str = "codex_responses",
) -> dict[str, Any]:
    return sa.build_session_owner_attributes(
        provider=provider,
        model=model,
        route_family=route_family,
        account_label=account,
        account_hash=f"hash-{account}",
        account_lane=f"{provider}:{account}:hash-{account}",
        endpoint_contract=endpoint_contract,
        state_format=state_format,
    )


async def _own_on_request(
    attrs: dict[str, Any],
    request: Any,
    *,
    session_id: str = SID,
    request_body: dict[str, Any] | None = None,
) -> Any:
    first = await sa.ensure_session_owner_guard_for_request(
        request=request,
        request_body=request_body or {"model": attrs.get("model")},
        session_identity=session_id,
        requested_attributes=attrs,
        require_exact_attributes=True,
    )
    assert first.held_reservation is True
    lease = sa.get_request_session_owner_lease(request)
    assert lease is not None
    await sa.finalize_session_owner_lease_on_success(lease, attributes=attrs)
    if hasattr(sa, "clear_non_held_request_session_owner_lease"):
        sa.clear_non_held_request_session_owner_lease(request)
    return first


@pytest.mark.asyncio
async def test_same_openai_hosted_provider_model_switch_with_previous_response_id_is_compatible_on_canonical_identity() -> None:
    redis = _FakeRedisCache()
    owner = _openai_attrs(model="gpt-5.6-sol", account="account1")
    requested = _openai_attrs(model="gpt-5.4", account="account1")
    request = _request()
    body = {"model": "gpt-5.4", "previous_response_id": "resp_prev"}
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ), patch.object(sa, "resolve_canonical_session_identity", return_value=SID):
        await _own_on_request(
            owner,
            request,
            request_body={"model": "gpt-5.6-sol"},
        )
        resume = _request()
        second = await sa.ensure_session_owner_guard_for_request(
            request=resume,
            request_body=body,
            session_identity=SID,
            requested_attributes=requested,
            require_exact_attributes=True,
            raise_on_redispatch=False,
        )
        assert second.decision is sa.SessionOwnerGuardDecision.COMPATIBLE_OWNER
        assert second.session_identity == SID
        assert sa.get_request_effective_session_identity(resume) is None


@pytest.mark.asyncio
async def test_openai_account_transfer_without_credential_affinity_is_compatible_on_canonical_identity() -> None:
    redis = _FakeRedisCache()
    owner = _openai_attrs(model="gpt-5.4", account="account1")
    requested = _openai_attrs(model="gpt-5.4", account="account2")
    assert "credential_affinity" not in owner
    assert "credential_affinity" not in requested
    request = _request()
    body = {"model": "gpt-5.4", "previous_response_id": "resp_prev"}
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ), patch.object(sa, "resolve_canonical_session_identity", return_value=SID):
        await _own_on_request(
            owner,
            request,
            request_body={"model": "gpt-5.4"},
        )
        resume = _request()
        second = await sa.ensure_session_owner_guard_for_request(
            request=resume,
            request_body=body,
            session_identity=SID,
            requested_attributes=requested,
            require_exact_attributes=True,
            raise_on_redispatch=False,
        )
        assert second.decision is sa.SessionOwnerGuardDecision.COMPATIBLE_OWNER
        assert second.session_identity == SID
        assert sa.get_request_effective_session_identity(resume) is None


@pytest.mark.asyncio
async def test_cross_provider_xai_switch_still_requires_redispatch() -> None:
    redis = _FakeRedisCache()
    owner = _openai_attrs(model="gpt-5.4", account="account1")
    requested = _provider_attrs(
        provider="xai",
        model="grok-4",
        account="account1",
    )
    request = _request()
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ), patch.object(sa, "resolve_canonical_session_identity", return_value=SID):
        await _own_on_request(
            owner,
            request,
            request_body={"model": "gpt-5.4"},
        )
        resume = _request()
        second = await sa.ensure_session_owner_guard_for_request(
            request=resume,
            request_body={"model": "grok-4", "previous_response_id": "resp_prev"},
            session_identity=SID,
            requested_attributes=requested,
            require_exact_attributes=True,
            raise_on_redispatch=False,
        )
        assert second.decision is sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED


@pytest.mark.asyncio
async def test_cross_provider_anthropic_switch_still_requires_redispatch() -> None:
    redis = _FakeRedisCache()
    owner = _openai_attrs(model="gpt-5.4", account="account1")
    requested = _provider_attrs(
        provider="anthropic",
        model="claude-sonnet-4",
        account="account1",
        route_family="anthropic_messages",
        endpoint_contract="anthropic_messages",
        state_format="anthropic_messages",
    )
    request = _request()
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ), patch.object(sa, "resolve_canonical_session_identity", return_value=SID):
        await _own_on_request(
            owner,
            request,
            request_body={"model": "gpt-5.4"},
        )
        resume = _request()
        second = await sa.ensure_session_owner_guard_for_request(
            request=resume,
            request_body={
                "model": "claude-sonnet-4",
                "previous_response_id": "resp_prev",
            },
            session_identity=SID,
            requested_attributes=requested,
            require_exact_attributes=True,
            raise_on_redispatch=False,
        )
        assert second.decision is sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED


@pytest.mark.asyncio
async def test_incompatible_openai_passthrough_endpoint_contract_still_requires_redispatch() -> None:
    redis = _FakeRedisCache()
    owner = _openai_attrs(model="gpt-5.4", account="account1")
    requested = _openai_attrs(
        model="gpt-5.4",
        account="account1",
        endpoint_contract="openai_passthrough",
        state_format="openai",
    )
    request = _request()
    with _patch_dual(redis), patch.object(
        durable_mod, "get_aawm_alias_routing_state_namespace", return_value="ns"
    ), patch.object(sa, "resolve_canonical_session_identity", return_value=SID):
        await _own_on_request(
            owner,
            request,
            request_body={"model": "gpt-5.4"},
        )
        resume = _request()
        second = await sa.ensure_session_owner_guard_for_request(
            request=resume,
            request_body={"model": "gpt-5.4", "previous_response_id": "resp_prev"},
            session_identity=SID,
            requested_attributes=requested,
            require_exact_attributes=True,
            raise_on_redispatch=False,
        )
        assert second.decision is sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED


def test_build_session_owner_id_treats_openai_model_and_account_as_mutable() -> None:
    owner = _openai_attrs(model="gpt-5.6-sol", account="account1")
    requested = _openai_attrs(model="gpt-5.4", account="account2")
    assert sa.build_session_owner_id(attributes=owner) == sa.build_session_owner_id(
        attributes=requested
    )


def test_compatibility_mismatch_reason_omits_model_from_hard_compare_tuple() -> None:
    source = inspect.getsource(sa._compatibility_mismatch_reason)
    tree = ast.parse(source)
    found_model_in_hard_tuple = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Tuple):
            continue
        elts: list[str] = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                elts.append(elt.value)
        if elts == ["provider", "model", "route_family"] or (
            "provider" in elts and "route_family" in elts and "model" in elts
        ):
            found_model_in_hard_tuple = True
    assert found_model_in_hard_tuple is False


def _handler_functions_containing_activate(source: str) -> list[str]:
    tree = ast.parse(source)
    texts: list[str] = []

    def _collect(fn_node: ast.AST) -> None:
        text = ast.get_source_segment(source, fn_node) or ""
        if "activate_session_owner_redispatch_effective_identity" in text:
            texts.append(text)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _collect(node)
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _collect(child)
    if texts:
        return texts
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _collect(node)
    return texts


def test_handler_keeps_activate_for_other_mismatches_but_has_same_provider_compatible_path() -> None:
    source = HANDLER_PATH.read_text()
    fn_sources = _handler_functions_containing_activate(source)
    assert fn_sources, (
        "expected openai_passthrough_handler.py function containing "
        "activate_session_owner_redispatch_effective_identity"
    )
    combined = "\n".join(fn_sources)
    assert "activate_session_owner_redispatch_effective_identity" in combined
    has_compatible_path = (
        "COMPATIBLE_OWNER" in combined
        or "compatible_owner" in combined
        or "same_hosted_provider" in combined
        or "hosted_provider" in combined
    )
    assert has_compatible_path, (
        "same-provider previous_response_id switches must take a compatible "
        "path instead of activate_session_owner_redispatch_effective_identity"
    )
