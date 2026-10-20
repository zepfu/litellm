"""ALI-016: explicit Alibaba Codex routes stop on the real admission gate.

Each case enters through the shipped ``try_dispatch_codex_request`` and the
real ``_resolve_codex_alibaba_token_plan_adapter_model``. The gate under test
is ``alibaba_token_plan_direct_route_admission`` in
``aawm_alias_routing.selection``. Only the provider route handlers are
counted; quota evidence, the hashed account cooldown key, and the admission
decision stay real.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Optional

import pytest
from fastapi import HTTPException, Request, Response

from litellm.llms.alibaba_token_plan.chat.transformation import (
    ALIBABA_TOKEN_PLAN_ACCOUNT_QUOTA_COOLDOWN_KEY_PREFIX,
    ALIBABA_TOKEN_PLAN_SUBSCRIPTION_ID_ENV,
    alibaba_token_plan_account_quota_cooldown_key,
    resolve_alibaba_token_plan_subscription_identity,
)
from litellm.proxy.pass_through_endpoints import (
    llm_passthrough_endpoints as lpe,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    cooldown_state,
    durable,
    selection,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.policy import (
    CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    AliasRoutingStateManager,
)

_MODEL = "alibaba_token_plan/qwen3.8-max"
_ENDPOINT = "/v1/responses"
_IDENTITY = "a" * 64
_OTHER_IDENTITY = "b" * 64
_ENVIRONMENT = "ali-016-direct-route"
_PROVIDER_HANDLERS = (
    "_handle_codex_opencode_zen_adapter_route",
    "_handle_codex_opencode_go_adapter_route",
    "_handle_codex_nous_chat_completions_adapter_route",
    "_handle_codex_kimi_chat_completions_adapter_route",
    "_handle_codex_alibaba_token_plan_adapter_route",
    "_handle_codex_zai_coding_plan_adapter_route",
    "_handle_codex_nvidia_completion_adapter_route",
    "_handle_codex_auto_agent_alias_route",
)
_ALIBABA_HANDLER = "_handle_codex_alibaba_token_plan_adapter_route"


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": _ENDPOINT,
            "raw_path": _ENDPOINT.encode(),
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 0),
            "server": ("test", 80),
        }
    )


def _alibaba_observation(
    *,
    window: str,
    remaining_pct: float,
    observed_at: float,
    account_hash: str,
) -> dict[str, Any]:
    return {
        "provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
        "model": _MODEL,
        "account_hash": account_hash,
        "environment": _ENVIRONMENT,
        "quota_key": f"alibaba_token_plan_{window}:credits",
        "quota_period": window,
        "quota_type": "credits",
        "remaining_pct": remaining_pct,
        "observed_at": observed_at,
        "expected_reset_at": observed_at + 3600.0,
        "status": "fresh",
        "exhausted": remaining_pct <= 0,
        "source": "alibaba_token_plan_usage",
    }


def _record_observations(*observations: dict[str, Any]) -> None:
    selection.alias_routing_state.record_normalized_quota_observations(
        list(observations)
    )


class _HandlerCounts:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    def handler(self, name: str):
        async def _handle(**_kwargs: Any) -> Response:
            self.calls[name] = self.calls.get(name, 0) + 1
            return Response(content=b"{}", media_type="application/json")

        return _handle

    def count(self, name: str) -> int:
        return self.calls.get(name, 0)


async def _dispatch(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, _HandlerCounts]:
    """Call the shipped explicit-route entry with only handlers replaced."""

    counts = _HandlerCounts()
    for name in _PROVIDER_HANDLERS:
        monkeypatch.setattr(lpe, name, counts.handler(name))
    body = {"model": _MODEL, "input": "ali-016"}
    try:
        result = await lpe.try_dispatch_codex_request(
            endpoint=_ENDPOINT,
            request=_request(),
            request_body=body,
            prepared_request_body=body,
            fastapi_response=Response(),
            user_api_key_dict=SimpleNamespace(),
            target_url="https://chatgpt.com/backend-api/codex/responses",
            api_key=None,
            forward_headers=False,
        )
    except HTTPException as exc:
        return exc, counts
    return result, counts


def _assert_only_alibaba(counts: _HandlerCounts, *, expected: int) -> None:
    assert counts.count(_ALIBABA_HANDLER) == expected
    for name in _PROVIDER_HANDLERS:
        if name == _ALIBABA_HANDLER:
            continue
        assert counts.count(name) == 0, name


def _assert_alibaba_denial(result: Any, *, reason: str) -> None:
    assert isinstance(result, HTTPException)
    assert result.status_code == 429
    detail = result.detail
    assert isinstance(detail, dict)
    error = detail["error"]
    assert error["type"] == "rate_limit_error"
    assert error["code"] == reason
    assert error["provider"] == "alibaba_token_plan"
    assert error["model"] == _MODEL
    assert detail["provider"] == "alibaba_token_plan"
    assert detail["model"] == _MODEL
    assert detail["reason"] == reason
    assert detail["subscription_identity"] == _IDENTITY


@pytest.fixture
def isolated_alibaba_state(monkeypatch: pytest.MonkeyPatch):
    """Fresh observations and a memory-only cooldown manager for one identity."""

    evidence_globals = selection._alibaba_token_plan_quota_evidence.__globals__
    # install() rebinds these helpers, so module-attribute patches miss the
    # globals the shipped admission path reads.
    require_manager = (
        cooldown_state._get_codex_auto_agent_active_cooldown_state.__globals__[
            "_require_manager"
        ]
    )
    cooldown_globals = require_manager.__globals__
    previous_manager = cooldown_globals["_manager"]
    previous_environment = evidence_globals["_get_codex_quota_observation_environment"]
    previous_cache_override = durable._dual_cache_override
    previous_observations = dict(
        selection.alias_routing_state._normalized_quota_observations
    )
    manager = AliasRoutingStateManager()
    cooldown_globals["_manager"] = manager
    evidence_globals["_get_codex_quota_observation_environment"] = (
        lambda: _ENVIRONMENT
    )
    durable._dual_cache_override = lambda: None
    selection.alias_routing_state._normalized_quota_observations.clear()
    monkeypatch.delenv(ALIBABA_TOKEN_PLAN_SUBSCRIPTION_ID_ENV, raising=False)
    monkeypatch.delenv("LITELLM_ALIBABA_TOKEN_PLAN_SETTINGS_FILE", raising=False)
    try:
        yield manager
    finally:
        cooldown_globals["_manager"] = previous_manager
        evidence_globals["_get_codex_quota_observation_environment"] = (
            previous_environment
        )
        durable._dual_cache_override = previous_cache_override
        observations = selection.alias_routing_state._normalized_quota_observations
        observations.clear()
        observations.update(previous_observations)


def _bind_identity(monkeypatch: pytest.MonkeyPatch, identity: Optional[str]) -> None:
    """Bind the identity ``alibaba_token_plan_direct_route_admission`` reads.

    The env resolver hashes an instance code, so it cannot emit the contract
    hash ``"a" * 64``. A missing identity keeps that resolver.
    """

    if identity is None:
        monkeypatch.delenv(ALIBABA_TOKEN_PLAN_SUBSCRIPTION_ID_ENV, raising=False)
        assert resolve_alibaba_token_plan_subscription_identity() is None
        return
    admission_globals = (
        selection.alibaba_token_plan_direct_route_admission.__globals__
    )
    monkeypatch.setitem(
        admission_globals,
        "resolve_alibaba_token_plan_subscription_identity",
        lambda: identity,
    )


@pytest.mark.asyncio
async def test_fresh_exact_account_exhaustion_denies_before_alibaba_handler(
    monkeypatch: pytest.MonkeyPatch,
    isolated_alibaba_state: AliasRoutingStateManager,
) -> None:
    _bind_identity(monkeypatch, _IDENTITY)
    now = time.time()
    _record_observations(
        _alibaba_observation(
            window="5h",
            remaining_pct=0.0,
            observed_at=now,
            account_hash=_IDENTITY,
        ),
        _alibaba_observation(
            window="7d",
            remaining_pct=40.0,
            observed_at=now,
            account_hash=_IDENTITY,
        ),
    )
    evidence, windows = selection._alibaba_token_plan_quota_evidence(
        subscription_identity=_IDENTITY,
    )
    assert evidence is None
    assert windows
    assert windows[0]["exhausted"] is True
    assert windows[0]["quota_key"] == "alibaba_token_plan_5h:credits"
    assert now - windows[0]["observed_at"] < 900

    result, counts = await _dispatch(monkeypatch)

    _assert_alibaba_denial(result, reason="quota_exhausted")
    _assert_only_alibaba(counts, expected=0)


@pytest.mark.asyncio
async def test_active_account_cooldown_denies_before_alibaba_handler(
    monkeypatch: pytest.MonkeyPatch,
    isolated_alibaba_state: AliasRoutingStateManager,
) -> None:
    _bind_identity(monkeypatch, _IDENTITY)
    cooldown_key = alibaba_token_plan_account_quota_cooldown_key(_IDENTITY)
    assert cooldown_key == (
        f"{ALIBABA_TOKEN_PLAN_ACCOUNT_QUOTA_COOLDOWN_KEY_PREFIX}:{_IDENTITY}"
    )
    assert cooldown_key is not None
    assert cooldown_key.endswith(f":{_IDENTITY}")
    isolated_alibaba_state.codex.cooldown_until_monotonic_by_key[cooldown_key] = (
        time.monotonic() + 60.0
    )

    result, counts = await _dispatch(monkeypatch)

    _assert_alibaba_denial(result, reason="account_quota_cooldown")
    _assert_only_alibaba(counts, expected=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("observation_age_seconds", [None, 901.0])
async def test_stale_or_unknown_evidence_invokes_only_alibaba_handler(
    monkeypatch: pytest.MonkeyPatch,
    isolated_alibaba_state: AliasRoutingStateManager,
    observation_age_seconds: Optional[float],
) -> None:
    _bind_identity(monkeypatch, _IDENTITY)
    if observation_age_seconds is not None:
        observed_at = time.time() - observation_age_seconds
        _record_observations(
            _alibaba_observation(
                window="5h",
                remaining_pct=0.0,
                observed_at=observed_at,
                account_hash=_IDENTITY,
            )
        )
        evidence, windows = selection._alibaba_token_plan_quota_evidence(
            subscription_identity=_IDENTITY,
        )
        assert evidence is None
        assert windows == []

    result, counts = await _dispatch(monkeypatch)

    assert isinstance(result, Response)
    assert not isinstance(result, HTTPException)
    _assert_only_alibaba(counts, expected=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("identity_bound", [False, True])
async def test_mismatched_or_missing_identity_still_invokes_alibaba_handler(
    monkeypatch: pytest.MonkeyPatch,
    isolated_alibaba_state: AliasRoutingStateManager,
    identity_bound: bool,
) -> None:
    _bind_identity(monkeypatch, _IDENTITY if identity_bound else None)
    now = time.time()
    _record_observations(
        _alibaba_observation(
            window="5h",
            remaining_pct=0.0,
            observed_at=now,
            account_hash=_OTHER_IDENTITY,
        ),
        _alibaba_observation(
            window="7d",
            remaining_pct=0.0,
            observed_at=now,
            account_hash=_OTHER_IDENTITY,
        ),
    )

    result, counts = await _dispatch(monkeypatch)

    assert isinstance(result, Response)
    assert not isinstance(result, HTTPException)
    _assert_only_alibaba(counts, expected=1)
