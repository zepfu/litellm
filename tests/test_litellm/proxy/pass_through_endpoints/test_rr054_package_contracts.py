"""RR-054 package contract coverage for ``aawm_alias_routing``.

This file is the ordinary test surface for package contracts and runtime behavior
covered under the extracted package scope.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException, Response
from fastapi.responses import StreamingResponse

from litellm.proxy.pass_through_endpoints import aawm_alias_routing as package
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    adapter_config,
    adapter_driver,
    antigravity_oauth,
    durable,
    google_oauth,
    memory,
    oauth_token_cache,
    policy,
    provider_shaping,
    responses_finalize,
    retry,
    state,
    streaming,
    task_state,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    AliasFamilyState,
    AliasRoutingStateManager,
    MonotonicCooldownMap,
    alias_routing_state,
)

PACKAGE_DIR = Path(package.__file__).resolve().parent
PACKAGE_MODULES = tuple(
    sorted(
        {path.stem for path in PACKAGE_DIR.glob("*.py") if path.name != "__init__.py"}
    )
)


# ---------------------------------------------------------------------------
# Package import / public surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name", PACKAGE_MODULES)
def test_rr054_package_module_imports_and_is_under_package(module_name: str) -> None:
    mod = importlib.import_module(
        f"litellm.proxy.pass_through_endpoints.aawm_alias_routing.{module_name}"
    )
    assert Path(mod.__file__).resolve().parent == PACKAGE_DIR
    assert mod.__name__.endswith(f".aawm_alias_routing.{module_name}")


def test_rr054_package_init_reexports_layers() -> None:
    assert package.policy is policy
    assert package.memory is memory
    assert package.retry is retry
    assert package.state is state
    assert package.adapter_config is adapter_config
    assert package.adapter_driver is adapter_driver
    assert package.responses_finalize is responses_finalize
    assert package.streaming is streaming
    assert package.task_state is task_state
    assert package.oauth_token_cache is oauth_token_cache
    assert package.provider_shaping is provider_shaping
    assert package.alias_routing_state is alias_routing_state
    assert package.AliasRoutingStateManager is AliasRoutingStateManager
    assert set(package.__all__) >= {
        "policy",
        "state",
        "memory",
        "retry",
        "adapter_config",
        "adapter_driver",
        "streaming",
        "task_state",
    }


# ---------------------------------------------------------------------------
# policy contracts
# ---------------------------------------------------------------------------


def test_rr054_policy_cooldown_defaults_and_alias_tables() -> None:
    assert policy.CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS == 30.0
    assert policy.CODEX_AUTO_AGENT_DEFAULT_USAGE_LIMIT_COOLDOWN_SECONDS == 3 * 60 * 60.0
    assert "aawm-codex-agent-auto" in policy.CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS
    assert "aawm-code" in policy.CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS
    assert "aawm-anthropic-agent-auto" in policy.ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS
    assert "aawm-code-anthropic" in policy.ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS

    for candidate in policy.CODEX_AUTO_AGENT_CANDIDATES:
        assert {"provider", "model", "route_family", "last_resort"} <= set(candidate)
    last = policy.CODEX_AUTO_AGENT_CANDIDATES[-1]
    assert last["last_resort"] is True

    assert "gpt-5.3-codex" in policy.ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS or (
        len(policy.ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS) > 0
    )
    assert policy.CODEX_AUTO_AGENT_OPENROUTER_PROVIDER == "openrouter"
    assert "openrouter/cohere/north-mini-code:free" in policy.OPENROUTER_FREE_DAILY_QUOTA_MODELS


# ---------------------------------------------------------------------------
# memory contracts
# ---------------------------------------------------------------------------


def test_rr054_memory_bound_and_cooldown_hydration_max_semantics() -> None:
    cache: dict[str, int] = {str(i): i for i in range(5)}
    memory.bound_memory_map(cache, max_size=3)
    assert list(cache.keys()) == ["2", "3", "4"]

    cooldown: dict[str, float] = {}
    future_epoch = time.time() + 120.0
    memory.hydrate_cooldown_memory(
        memory_map=cooldown,
        cooldown_key="lane-a",
        expires_at_epoch=future_epoch,
        max_size=10,
    )
    first_until = cooldown["lane-a"]
    # Shorter residual must not shrink an existing longer cooldown.
    memory.hydrate_cooldown_memory(
        memory_map=cooldown,
        cooldown_key="lane-a",
        expires_at_epoch=time.time() + 5.0,
        max_size=10,
    )
    assert cooldown["lane-a"] == first_until
    assert memory.remaining_cooldown_seconds(cooldown, "lane-a") > 0


def test_rr054_memory_affinity_hydration_preserves_fresher_local() -> None:
    affinity_map: dict[str, dict[str, Any]] = {}
    local_until = time.monotonic() + 500.0
    affinity_map["sess"] = {
        "provider": "local",
        "model": "local-model",
        "route_family": "local-family",
        "last_resort": False,
        "expires_at_monotonic": local_until,
    }
    # Durable payload with shorter remaining must not clobber fresher local.
    result = memory.hydrate_affinity_memory(
        memory_map=affinity_map,
        session_key="sess",
        payload={
            "provider": "redis",
            "model": "redis-model",
            "route_family": "redis-family",
            "last_resort": True,
        },
        expires_at_epoch=time.time() + 10.0,
        max_size=10,
    )
    assert result["provider"] == "local"
    assert affinity_map["sess"]["provider"] == "local"

    # Empty / expired durable payload is a no-op.
    empty = memory.hydrate_affinity_memory(
        memory_map={},
        session_key="other",
        payload={"provider": "x", "model": "y", "route_family": "z"},
        expires_at_epoch=time.time() - 1.0,
    )
    assert empty == {}


def test_rr054_memory_extend_and_max_remaining() -> None:
    m: dict[str, float] = {}
    until = memory.extend_monotonic_cooldown(m, "k", 2.0, max_size=8)
    assert until > time.monotonic()
    # shorter wait does not shrink
    again = memory.extend_monotonic_cooldown(m, "k", 0.1, max_size=8)
    assert again == until
    assert memory.max_remaining_cooldown_seconds(m, ["k", "missing"]) > 0
    assert memory.max_remaining_cooldown_seconds(m, []) == 0.0


# ---------------------------------------------------------------------------
# state contracts
# ---------------------------------------------------------------------------


def test_rr054_state_family_cooldown_and_negative_cache() -> None:
    family = AliasFamilyState()
    family.set_cooldown_memory("c1", 5.0, max_size=16)
    assert family.get_memory_cooldown_remaining("c1") > 0
    family.mark_negative_cache("c1", ttl_seconds=2.0, max_size=16)
    assert family.is_negative_cached("c1") is True
    # A longer cooldown write clears negative cache; a shorter one is a no-op.
    family.set_cooldown_memory("c1", 1.0, max_size=16)
    assert family.is_negative_cached("c1") is True
    family.set_cooldown_memory("c1", 10.0, max_size=16)
    assert family.is_negative_cached("c1") is False
    family.mark_negative_cache("neg", ttl_seconds=3.0, max_size=16)
    assert family.is_negative_cached("neg") is True
    family.clear_negative_cache("neg")
    assert family.is_negative_cached("neg") is False

    remaining = family.hydrate_cooldown("c2", time.time() + 30.0, max_size=16)
    assert remaining > 0


def test_rr054_state_affinity_memory_roundtrip_and_expiry() -> None:
    family = AliasFamilyState()
    family.set_affinity_memory(
        "sess-1",
        {
            "provider": "openai",
            "model": "gpt",
            "route_family": "codex",
            "last_resort": False,
        },
        ttl_seconds=60.0,
        max_size=16,
    )
    got = family.get_affinity_memory("sess-1")
    assert got is not None
    assert got["provider"] == "openai"
    assert got["affinity_state_source"] == "memory"

    family.session_affinity_by_key["expired"] = {
        "provider": "x",
        "model": "y",
        "route_family": "z",
        "last_resort": False,
        "expires_at_monotonic": time.monotonic() - 1.0,
    }
    assert family.get_affinity_memory("expired") is None
    assert "expired" not in family.session_affinity_by_key

    hydrated = family.hydrate_affinity(
        "sess-2",
        {
            "provider": "openrouter",
            "model": "free",
            "route_family": "or",
            "last_resort": True,
        },
        expires_at_epoch=time.time() + 45.0,
        max_size=16,
    )
    assert hydrated["provider"] == "openrouter"


def test_rr054_state_manager_families_and_monotonic_maps() -> None:
    manager = AliasRoutingStateManager(max_size=32)
    assert manager.family("anthropic") is manager.anthropic
    assert manager.family("codex") is manager.codex
    assert manager.family("other") is manager.codex
    assert manager.google_oauth is oauth_token_cache.google_oauth_access_token_cache
    assert (
        manager.antigravity_oauth
        is oauth_token_cache.antigravity_oauth_access_token_cache
    )

    cmap = MonotonicCooldownMap()
    cmap.extend("r1", 1.5, max_size=8)
    assert cmap.remaining("r1") > 0
    assert cmap.max_remaining(["r1", "missing"]) > 0
    assert cmap.max_remaining([]) == 0.0

    assert isinstance(alias_routing_state, AliasRoutingStateManager)
    assert alias_routing_state.codex is not alias_routing_state.anthropic


@pytest.mark.asyncio
async def test_rr054_state_candidate_probe_lock_is_stable_per_key() -> None:
    manager = AliasRoutingStateManager()
    first = await manager.candidate_probe_lock(
        alias_family="codex", cooldown_key="lane:1"
    )
    second = await manager.candidate_probe_lock(
        alias_family="codex", cooldown_key="lane:1"
    )
    other = await manager.candidate_probe_lock(
        alias_family="codex", cooldown_key="lane:2"
    )
    assert first is second
    assert first is not other
    assert isinstance(first, asyncio.Lock)


# ---------------------------------------------------------------------------
# durable contracts
# ---------------------------------------------------------------------------


def test_rr054_durable_cache_key_and_expiry_parse() -> None:
    durable.configure_durable_runtime(clean_value=lambda v: str(v).strip() if v else None)
    key = durable.build_aawm_alias_routing_durable_cache_key(
        alias_family="Codex",
        state_kind="Cooldown",
        state_key="session-or-lane",
    )
    assert key.startswith(f"{durable.AAWM_ALIAS_ROUTING_STATE_KEY_PREFIX}:")
    assert ":codex:cooldown:" in key
    # opaque hashed state key (not raw)
    assert "session-or-lane" not in key

    assert durable.parse_aawm_alias_routing_durable_expiry("nope") is None
    assert durable.parse_aawm_alias_routing_durable_expiry({}) is None
    assert (
        durable.parse_aawm_alias_routing_durable_expiry(
            {"expires_at_epoch": time.time() - 5}
        )
        is None
    )
    future = time.time() + 100
    assert durable.parse_aawm_alias_routing_durable_expiry(
        {"expires_at_epoch": future}
    ) == pytest.approx(future)


def test_rr054_durable_namespace_and_failure_log_throttle() -> None:
    ns = durable.get_aawm_alias_routing_state_namespace()
    assert isinstance(ns, str) and ns
    # throttle map starts allowing first log
    assert durable._should_log_durable_failure("unit-test-key-a") is True
    assert durable._should_log_durable_failure("unit-test-key-a") is False


@pytest.mark.asyncio
async def test_rr054_durable_read_write_max_expiry_and_missing_cache() -> None:
    # no dual cache => write/read fail closed
    with patch.object(durable, "get_aawm_alias_routing_dual_cache", return_value=None):
        assert (
            await durable.read_aawm_alias_routing_durable_payload(
                alias_family="codex",
                state_kind="cooldown",
                state_key="k",
            )
            is None
        )
        assert (
            await durable.write_aawm_alias_routing_durable_payload(
                alias_family="codex",
                state_kind="cooldown",
                state_key="k",
                payload={"cooldown_key": "k"},
                ttl_seconds=30.0,
            )
            is False
        )

    dual = MagicMock()
    dual.redis_cache = MagicMock()
    dual.redis_cache.async_set_cache = AsyncMock(return_value=None)
    dual.async_set_cache = AsyncMock(return_value=None)
    existing_expires = time.time() + 3600.0
    dual.async_get_cache = AsyncMock(
        return_value={"cooldown_key": "k", "expires_at_epoch": existing_expires, "extra": 1}
    )

    with patch.object(durable, "get_aawm_alias_routing_dual_cache", return_value=dual), patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable.get_durable_write_retry_attempts",
        return_value=0,
    ), patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable.get_durable_write_retry_backoff_seconds",
        return_value=0.0,
    ):
        ok = await durable.write_aawm_alias_routing_durable_payload(
            alias_family="codex",
            state_kind="cooldown",
            state_key="k",
            payload={"cooldown_key": "k", "reason": "transient"},
            ttl_seconds=30.0,
        )
    assert ok is True
    written = dual.redis_cache.async_set_cache.await_args.kwargs["value"]
    assert written["expires_at_epoch"] == pytest.approx(existing_expires, abs=1.0)
    assert written["reason"] == "transient"
    assert written["extra"] == 1

    dual.async_get_cache = AsyncMock(
        return_value={"expires_at_epoch": time.time() + 50, "ok": True}
    )
    with patch.object(durable, "get_aawm_alias_routing_dual_cache", return_value=dual):
        payload = await durable.read_aawm_alias_routing_durable_payload(
            alias_family="codex",
            state_kind="cooldown",
            state_key="k",
        )
    assert payload is not None and payload["ok"] is True


# ---------------------------------------------------------------------------
# oauth token cache + pure oauth helpers
# ---------------------------------------------------------------------------


def test_rr054_oauth_token_cache_get_set_clear() -> None:
    cache = oauth_token_cache.OAuthAccessTokenCache()
    now = time.time()
    # expiry stored as millis by default
    cache.set("lane", "tok-1", int((now + 120) * 1000))
    assert cache.get_if_valid("lane", now=now, skew_seconds=30.0) == "tok-1"
    cache.set("lane", "tok-2", int((now + 10) * 1000))
    assert cache.get_if_valid("lane", now=now, skew_seconds=30.0) is None
    cache.set("lane", "tok-3", int(now + 120))  # seconds mode
    assert (
        cache.get_if_valid("lane", now=now, skew_seconds=1.0, expiry_is_millis=False)
        == "tok-3"
    )
    cache.clear("lane")
    assert cache.get_if_valid("lane", now=now) is None
    cache.set("a", "t", int((now + 60) * 1000))
    cache.clear()
    assert cache.tokens == {}


def test_rr054_google_oauth_pure_validation_helpers() -> None:
    google_oauth.configure_google_oauth_runtime(
        clean_value=google_oauth._default_clean,
        get_first_secret_value=lambda _names: None,
        invalidate_google_lane_cache=lambda: None,
    )
    assert google_oauth._default_clean("  x  ") == "x"
    assert google_oauth._default_clean("   ") is None
    assert google_oauth._default_clean(None) is None

    future_ms = int((time.time() + 120) * 1000)
    past_ms = int((time.time() - 5) * 1000)
    assert (
        google_oauth._google_oauth_token_is_valid(
            {"access_token": "abc", "expiry_date": future_ms}
        )
        is True
    )
    assert (
        google_oauth._google_oauth_token_is_valid(
            {"access_token": "abc", "expiry_date": past_ms}
        )
        is False
    )
    assert google_oauth._google_oauth_token_is_valid({"access_token": ""}) is False
    assert (
        google_oauth._google_oauth_cached_token_is_valid(("tok", future_ms)) is True
    )
    assert google_oauth._get_google_oauth_expiry_date({"expiry_date": 12.5}) == 12
    assert google_oauth._get_google_oauth_expiry_date({}) is None
    assert (
        google_oauth._get_google_oauth_client_value(
            {"client_id": " from-auth "},
            ("client_id",),
            ("MISSING_ENV",),
        )
        == "from-auth"
    )


def test_rr054_antigravity_oauth_pure_validation_helpers() -> None:
    assert antigravity_oauth._default_clean(" a ") == "a"
    future = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    valid = {"token": {"access_token": "tok", "expiry": future}}
    expired = {"token": {"access_token": "tok", "expiry": past}}
    assert antigravity_oauth._parse_antigravity_token_expiry(future) is not None
    assert antigravity_oauth._parse_antigravity_token_expiry("not-a-date") is None
    assert antigravity_oauth._antigravity_access_token_is_valid(valid) is True
    assert antigravity_oauth._antigravity_access_token_is_valid(expired) is False
    assert antigravity_oauth._antigravity_access_token_is_unexpired(valid) is True
    expiry_ms = antigravity_oauth._get_antigravity_oauth_expiry_date(valid)
    assert isinstance(expiry_ms, int) and expiry_ms > 0
    assert (
        antigravity_oauth._antigravity_oauth_cached_token_is_valid(
            ("tok", int((time.time() + 120) * 1000))
        )
        is True
    )

    # Package fallback is safe without a response; god-file import may have
    # already installed production hooks, so isolate both contract branches.
    previous_format = antigravity_oauth._format_refresh_failure_fn
    try:
        antigravity_oauth._format_refresh_failure_fn = None
        msg = antigravity_oauth._format_refresh_failure(
            provider_label="Antigravity", response=None
        )
        assert "Antigravity" in msg
        assert "Failed to refresh" in msg

        # Configured production-shaped contract requires a real HTTP response
        # with status_code + .json() (httpx.Response).
        def _format(*, provider_label: str, response: Any) -> str:
            body = response.json()
            code = body.get("error") if isinstance(body, dict) else None
            return (
                f"Failed to refresh {provider_label} OAuth access token "
                f"(status={response.status_code}, error={code})."
            )

        antigravity_oauth._format_refresh_failure_fn = _format
        response = httpx.Response(
            401,
            json={"error": "invalid_grant"},
            request=httpx.Request("POST", "https://oauth.example.test/token"),
        )
        configured = antigravity_oauth._format_refresh_failure(
            provider_label="Antigravity",
            response=response,
        )
        assert "status=401" in configured
        assert "invalid_grant" in configured
    finally:
        antigravity_oauth._format_refresh_failure_fn = previous_format


# ---------------------------------------------------------------------------
# retry contracts
# ---------------------------------------------------------------------------


def test_rr054_retry_normalize_keys_and_env_parsers() -> None:
    assert retry.normalize_cooldown_keys("only") == ["only"]
    assert retry.normalize_cooldown_keys(["a", "", "b", 1]) == ["a", "b"]  # type: ignore[list-item]
    assert retry.normalize_cooldown_keys(None) == ["__default__"]
    assert retry.normalize_cooldown_keys([]) == ["__default__"]

    env = {"F": "1.5", "I": "3", "BAD": "x", "NEG": "-2", "BIG": "99"}
    assert (
        retry.parse_non_negative_float_env(
            "F", default=0.0, getenv=env.get, maximum=10.0
        )
        == 1.5
    )
    assert retry.parse_non_negative_float_env("BAD", default=9.0, getenv=env.get) == 9.0
    assert (
        retry.parse_non_negative_float_env(
            "NEG", default=1.0, minimum=0.0, getenv=env.get
        )
        == 0.0
    )
    assert (
        retry.parse_non_negative_int_env("I", default=0, getenv=env.get, maximum=10)
        == 3
    )
    assert retry.parse_non_negative_int_env("BAD", default=4, getenv=env.get) == 4
    assert (
        retry.parse_non_negative_int_env(
            "BIG", default=1, maximum=5, getenv=env.get
        )
        == 5
    )

    projected, within = retry.projected_hidden_retry_within_budget(
        accumulated_hidden_wait_seconds=2.0,
        next_wait_seconds=3.0,
        hidden_retry_budget_seconds=10.0,
    )
    assert projected == 5.0 and within is True
    _, over = retry.projected_hidden_retry_within_budget(
        accumulated_hidden_wait_seconds=8.0,
        next_wait_seconds=5.0,
        hidden_retry_budget_seconds=10.0,
    )
    assert over is False
    delay = retry.exponential_backoff_seconds(
        3, base_seconds=1.0, max_seconds=10.0, jitter_seconds=0.0
    )
    assert delay == 4.0


@pytest.mark.asyncio
async def test_rr054_retry_wait_set_and_policy_loop() -> None:
    cmap = MonotonicCooldownMap()
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    await retry.set_monotonic_cooldown_map(cmap, ["k1", "k2"], 0.25, max_size=8)
    waited = await retry.wait_for_monotonic_cooldown_map(
        cmap,
        ["k1", "k2"],
        log_label="unit-test",
        sleep=fake_sleep,
    )
    assert waited > 0
    assert slept and slept[0] == pytest.approx(waited, abs=0.05)

    attempts: list[int] = []

    async def before(attempt: int) -> None:
        attempts.append(attempt)

    async def on_failure(_exc: Exception, attempt: int) -> bool:
        return attempt < 2

    async def op() -> str:
        if len(attempts) < 2:
            raise RuntimeError("transient")
        return "ok"

    result = await retry.run_adapter_retry_policy(
        op,
        policy=retry.AdapterRetryPolicy(before_attempt=before, on_failure=on_failure),
    )
    assert result == "ok"
    assert attempts == [1, 2]


# ---------------------------------------------------------------------------
# adapter config / driver
# ---------------------------------------------------------------------------


def test_rr054_adapter_config_descriptors_and_finalize_kwargs() -> None:
    assert adapter_config.OPENAI_RESPONSES.adapter == (
        "anthropic_openai_responses_adapter"
    )
    assert adapter_config.OPENAI_RESPONSES.default_use_codex_native_tools is True
    assert adapter_config.OPENROUTER_RESPONSES.reject_empty_success is True
    assert adapter_config.NVIDIA_COMPLETION.custom_llm_provider == "nvidia_nim"

    kwargs = adapter_config.responses_finalize_kwargs(
        adapter_config.OPENROUTER_RESPONSES,
        adapter_model="or-model",
        translated_request_body={"model": "req", "stream": False},
    )
    assert kwargs["adapter"] == adapter_config.OPENROUTER_RESPONSES.adapter
    assert kwargs["response_builder_kwargs"]["reject_empty_success"] is True
    assert (
        kwargs["response_builder_kwargs"]["diagnostic_context"]["adapter_model"]
        == "or-model"
    )

    openai_kwargs = adapter_config.responses_finalize_kwargs(
        adapter_config.OPENAI_RESPONSES,
        adapter_model="o",
        translated_request_body={},
    )
    assert "response_builder_kwargs" not in openai_kwargs
    assert openai_kwargs["use_codex_native_tools"] is True


@pytest.mark.asyncio
async def test_rr054_adapter_driver_runs_prepare_then_perform_and_handles_errors() -> None:
    plan = adapter_driver.ResponsesAdapterRoutePlan(
        config=adapter_config.OPENAI_RESPONSES,
        translated_request_body={"model": "m"},
        target_url="https://example.test/v1",
        custom_headers={"h": "1"},
        client_requested_stream=False,
        perform_kwargs={"extra": True},
    )

    async def prepare(**_kwargs: Any) -> adapter_driver.ResponsesAdapterRoutePlan:
        return plan

    seen: dict[str, Any] = {}

    async def perform(**kwargs: Any) -> str:
        seen.update(kwargs)
        return "done"

    out = await adapter_driver.run_responses_adapter_route(
        prepare=prepare,
        perform=perform,
        request=object(),
        user_api_key_dict=object(),
        prepared_request_body={"x": 1},
        adapter_model="m",
        use_alias_candidate_probe=False,
    )
    assert out == "done"
    assert seen["config"] is adapter_config.OPENAI_RESPONSES
    assert seen["extra"] is True

    handled: list[Exception] = []

    def handle(exc: Exception) -> None:
        handled.append(exc)

    fail_plan = adapter_driver.CompletionAdapterRoutePlan(
        config=adapter_config.OPENROUTER_COMPLETION,
        prepared_request_body={"m": 1},
        target_url="https://example.test/chat",
        api_key="k",
        api_base="https://example.test",
        client_requested_stream=False,
        handle_exception=handle,
    )

    async def prepare_completion(
        **_kwargs: Any,
    ) -> adapter_driver.CompletionAdapterRoutePlan:
        return fail_plan

    async def boom(**_kwargs: Any) -> Any:
        raise ValueError("upstream")

    with pytest.raises(ValueError, match="upstream"):
        await adapter_driver.run_completion_adapter_route(
            prepare=prepare_completion,
            perform=boom,
            request=object(),
            prepared_request_body={},
            adapter_model="m",
            use_alias_candidate_probe=True,
        )
    assert len(handled) == 1


# ---------------------------------------------------------------------------
# provider shaping / task state / streaming / responses finalize
# ---------------------------------------------------------------------------


def test_rr054_provider_shaping_json_prefix_and_delimited_spans() -> None:
    payload, end = provider_shaping.decode_json_prefix('  {"a": 1} trailing')
    assert json.loads(payload) == {"a": 1}
    assert end > 0

    with pytest.raises(json.JSONDecodeError):
        provider_shaping.decode_json_prefix("not-json")

    transformed_payload, end2 = provider_shaping.decode_json_prefix(
        "x{\"a\": 1}",
        fallback_transform=lambda s: s[1:] if s.startswith("x") else s,
    )
    assert json.loads(transformed_payload) == {"a": 1}
    assert end2 > 0

    text = "pre <sys>one</sys>\nmid <sys>two</sys> tail"
    spans = provider_shaping.iter_delimited_spans(text, "<sys>", "</sys>")
    assert len(spans) == 2
    assert text[spans[0].start : spans[0].end].startswith("<sys>")
    assert provider_shaping.iter_delimited_spans("nope", "<a>", "</a>") == []
    # unclosed opener does not hang / restart
    assert provider_shaping.iter_delimited_spans("<sys>open", "<sys>", "</sys>") == []


def test_rr054_task_state_markers_and_selection() -> None:
    assert task_state.resolve_task_state_markers("") == task_state.DEFAULT_TASK_STATE_MARKERS
    custom = task_state.resolve_task_state_markers("alpha, beta\ngamma")
    assert custom == ("alpha", "beta", "gamma")

    assert (
        task_state.message_has_structured_task_state_flag(
            {"metadata": {"preserve_task_state": True}}
        )
        is True
    )
    assert (
        task_state.message_has_structured_task_state_flag({"task_state": "yes"}) is True
    )
    assert task_state.message_has_structured_task_state_flag({"content": "x"}) is False

    messages = [
        {"role": "user", "content": "noise"},
        {
            "role": "user",
            "content": "please continue the task carefully",
            "metadata": {},
        },
        {"role": "user", "content": "acceptance criteria listed here"},
    ]

    def extract_text(message: dict[str, Any]) -> str:
        return str(message.get("content") or "")

    def is_skippable(message: dict[str, Any]) -> bool:
        return message.get("role") == "system"

    selected = task_state.select_task_state_source(
        messages,
        extract_text=extract_text,
        is_skippable=is_skippable,
    )
    assert selected is not None
    index, text, kind = selected
    assert kind in {"marker", "fallback", "structured"}
    assert index >= 0 and text

    structured = task_state.select_task_state_source(
        [
            {"role": "user", "content": "first", "metadata": {"task_state": True}},
            {"role": "user", "content": "continue the task"},
        ],
        extract_text=extract_text,
        is_skippable=is_skippable,
    )
    assert structured is not None and structured[2] == "structured"


@pytest.mark.asyncio
async def test_rr054_streaming_peek_exhausted_and_overflow() -> None:
    async def small_body() -> Any:
        yield b"a"
        yield b"b"

    small = StreamingResponse(small_body(), media_type="text/event-stream", status_code=200)
    peek = await streaming.peek_streaming_response(small, max_chunks=10, max_bytes=100)
    assert peek.exhausted is True
    assert peek.stop_reason == "stream_exhausted"
    assert peek.buffered_chunks == [b"a", b"b"]
    replayed: list[Any] = []
    async for chunk in peek.response.body_iterator:
        replayed.append(chunk)
    assert replayed == [b"a", b"b"]

    async def large_body() -> Any:
        yield b"12345"
        yield b"67890"
        yield b"more"

    large = StreamingResponse(large_body(), media_type="text/event-stream", status_code=200)
    peek2 = await streaming.peek_streaming_response(large, max_chunks=1, max_bytes=1000)
    assert peek2.exhausted is False
    assert peek2.stop_reason == "chunk_limit"
    # lossless continuation still yields all bytes
    cont: list[Any] = []
    async for chunk in peek2.response.body_iterator:
        cont.append(chunk)
    assert cont[0] == b"12345"
    assert b"more" in cont or cont[-1] == b"more"


def _minimal_responses_api_payload(*, response_id: str = "resp_1") -> dict[str, Any]:
    """Minimal valid Responses API body used by package finalize contracts."""
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "test-model",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "hello from adapter",
                        "annotations": [],
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
        },
    }


@pytest.mark.asyncio
async def test_rr054_responses_finalize_nonstream_and_requires_runtime() -> None:
    # Isolate package runtime from god-file configure side effects.
    previous_runtime = responses_finalize._runtime
    try:
        responses_finalize._runtime = None
        with pytest.raises(RuntimeError, match="not been configured"):
            await responses_finalize.finalize_anthropic_responses_adapter_upstream_response(
                upstream_response=Response(
                    content=json.dumps(_minimal_responses_api_payload()).encode("utf-8"),
                    media_type="application/json",
                ),
                request=SimpleNamespace(),
                translated_request_body={"model": "m", "stream": False},
                adapter_model="m",
                adapter="a",
                adapter_label="A",
                provider="openai",
                target_url="https://example.test",
                client_requested_stream=False,
                use_alias_candidate_probe=False,
                unexpected_detail="bad",
            )

        annotated: list[Any] = []
        captured_bodies: list[dict[str, Any]] = []

        def annotate(request: Any, target_url: Any) -> None:
            annotated.append((request, target_url))

        def build_response(body: dict[str, Any], **_kwargs: Any) -> Response:
            # Package finalize only decodes JSON and delegates to runtime builder.
            # Keep a pure capture builder so this package contract stays hermetic
            # even if production builder enforces ResponsesAPIResponse fields.
            captured_bodies.append(body)
            return Response(
                content=json.dumps({"ok": body.get("id")}).encode("utf-8"),
                media_type="application/json",
            )

        def copy_headers(
            *, translated_response: Response, upstream_response: Response
        ) -> None:
            translated_response.headers["x-copied"] = upstream_response.headers.get(
                "x-up", "1"
            )

        def decode(body: Any) -> str:
            return bytes(body).decode("utf-8")

        def malformed(**_kwargs: Any) -> dict[str, Any]:
            return {"malformed": True}

        responses_finalize.configure_responses_finalize_runtime(
            responses_finalize.ResponsesFinalizeRuntime(
                annotate_request=annotate,
                validate_stream=AsyncMock(),
                collect_stream=AsyncMock(),
                build_response=build_response,
                copy_headers=copy_headers,
                build_streaming_response=lambda *a, **k: StreamingResponse(iter(())),
                decode_response_body=decode,
                build_malformed_context=malformed,
            )
        )
        payload = _minimal_responses_api_payload(response_id="resp_1")
        upstream = Response(
            content=json.dumps(payload).encode("utf-8"),
            status_code=201,
            headers={"x-up": "yes"},
            media_type="application/json",
        )
        out = await responses_finalize.finalize_anthropic_responses_adapter_upstream_response(
            upstream_response=upstream,
            request=SimpleNamespace(headers={}),
            translated_request_body={"model": "m", "stream": False},
            adapter_model="m",
            adapter="anthropic_openai_responses_adapter",
            adapter_label="OpenAI",
            provider="openai",
            target_url="https://example.test/v1",
            client_requested_stream=False,
            use_alias_candidate_probe=False,
            unexpected_detail="Unexpected upstream",
        )
        assert out.status_code == 201
        assert json.loads(out.body.decode("utf-8"))["ok"] == "resp_1"
        assert annotated
        assert captured_bodies and captured_bodies[0]["id"] == "resp_1"
        assert captured_bodies[0]["status"] == "completed"
        assert captured_bodies[0]["output"]
        assert out.headers.get("x-copied") == "yes"

        with pytest.raises(HTTPException) as excinfo:
            await responses_finalize.finalize_anthropic_responses_adapter_upstream_response(
                upstream_response=object(),
                request=SimpleNamespace(),
                translated_request_body={"model": "m"},
                adapter_model="m",
                adapter="a",
                adapter_label="A",
                provider="openai",
                target_url="https://example.test",
                client_requested_stream=False,
                use_alias_candidate_probe=False,
                unexpected_detail="Unexpected upstream",
            )
        assert excinfo.value.status_code == 502
    finally:
        responses_finalize._runtime = previous_runtime
