"""Wave 5C module-local tests for cooldown_apply.py.

Exercises the extracted cooldown-publication-plan resolver and applicators
directly with injected seams -- no ambient god-module state, no import of
``llm_passthrough_endpoints`` at module scope.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import cooldown_apply
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_apply import (
    _apply_anthropic_auto_agent_alias_cooldown,
    _apply_auto_agent_alias_cooldown,
    _apply_codex_auto_agent_alias_cooldown,
    _apply_basic_pilot_gated_cooldown,
    _persist_anthropic_cooldown_durable,
    _persist_codex_cooldown_durable,
    _resolve_auto_agent_cooldown_publication_plan,
    _set_codex_auto_agent_candidate_cooldowns,
    configure_cooldown_apply_runtime,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.interfaces import (
    CooldownPublicationPlan,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    AliasRoutingStateManager,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeDecision:
    should_cool: bool = False
    duration_seconds: float = 0.0
    scope: Optional[str] = None


class _FakeGate:
    def __init__(self, decision: _FakeDecision) -> None:
        self._decision = decision

    def current_decision(self, *, cooldown_key: str, **kwargs: Any) -> _FakeDecision:
        return self._decision


def _make_request() -> MagicMock:
    req = MagicMock()
    req.state = MagicMock()
    return req


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def configured_runtime():
    """Configure cooldown_apply with controllable stubs and restore after."""
    # Save previous seam values
    prev = {
        name: getattr(cooldown_apply, name)
        for name in (
            "_get_candidate_cooldown_scope",
            "_get_kimi_managed_account_cooldown_key",
            "_get_grok_account_quota_lane_cooldown_key",
            "_get_request_local_cooldown_key",
            "_set_request_local_cooldown",
            "_exclude_request_local_candidate",
            "_set_codex_cooldown",
            "_set_anthropic_cooldown",
            "_write_durable_payload",
            "_basic_pilot_gate",
            "_state_manager",
        )
    }
    previous_host_globals = cooldown_apply._host_globals_ref
    missing = object()
    previous_host_runtime = (
        {
            name: previous_host_globals.get(name, missing)
            for name in prev
        }
        if previous_host_globals is not None
        else {}
    )

    scope_fn = MagicMock(return_value="none")
    kimi_key_fn = MagicMock(return_value="kimi:__managed__:default")
    grok_lane_fn = MagicMock(return_value=None)
    rl_key_fn = MagicMock(return_value="rl:key")
    rl_set_fn = MagicMock()
    rl_exclude_fn = MagicMock()
    codex_set = AsyncMock()
    anthropic_set = AsyncMock()
    write_durable = AsyncMock()
    gate = _FakeGate(_FakeDecision())
    mgr = AliasRoutingStateManager()

    configure_cooldown_apply_runtime(
        get_candidate_cooldown_scope=scope_fn,
        get_kimi_managed_account_cooldown_key=kimi_key_fn,
        get_grok_account_quota_lane_cooldown_key=grok_lane_fn,
        get_request_local_cooldown_key=rl_key_fn,
        set_request_local_cooldown=rl_set_fn,
        exclude_request_local_candidate=rl_exclude_fn,
        set_codex_cooldown=codex_set,
        set_anthropic_cooldown=anthropic_set,
        write_durable_payload=write_durable,
        basic_pilot_gate=gate,
        state_manager=mgr,
    )

    yield {
        "scope_fn": scope_fn,
        "kimi_key_fn": kimi_key_fn,
        "grok_lane_fn": grok_lane_fn,
        "rl_key_fn": rl_key_fn,
        "rl_set_fn": rl_set_fn,
        "rl_exclude_fn": rl_exclude_fn,
        "codex_set": codex_set,
        "anthropic_set": anthropic_set,
        "write_durable": write_durable,
        "gate": gate,
        "mgr": mgr,
    }

    # Restore
    for name, val in prev.items():
        setattr(cooldown_apply, name, val)
    if previous_host_globals is not None:
        for name, val in previous_host_runtime.items():
            if val is missing:
                previous_host_globals.pop(name, None)
            else:
                previous_host_globals[name] = val
    cooldown_apply._host_globals_ref = previous_host_globals


def test_install_then_configure_keeps_host_bindings_coherent(
    configured_runtime: dict[str, Any],
) -> None:
    previous_host_globals = cooldown_apply._host_globals_ref
    status_sentinel = object()
    http_exception_sentinel = object()
    logger_sentinel = object()
    host_globals = {
        "status": status_sentinel,
        "HTTPException": http_exception_sentinel,
        "verbose_proxy_logger": logger_sentinel,
    }
    try:
        cooldown_apply.install(host_globals)
        configure_cooldown_apply_runtime(
            get_candidate_cooldown_scope=configured_runtime["scope_fn"],
            get_kimi_managed_account_cooldown_key=configured_runtime["kimi_key_fn"],
            get_grok_account_quota_lane_cooldown_key=configured_runtime["grok_lane_fn"],
            get_request_local_cooldown_key=configured_runtime["rl_key_fn"],
            set_request_local_cooldown=configured_runtime["rl_set_fn"],
            exclude_request_local_candidate=configured_runtime["rl_exclude_fn"],
            set_codex_cooldown=configured_runtime["codex_set"],
            set_anthropic_cooldown=configured_runtime["anthropic_set"],
            write_durable_payload=configured_runtime["write_durable"],
            basic_pilot_gate=configured_runtime["gate"],
            state_manager=configured_runtime["mgr"],
        )

        assert cooldown_apply._host_globals_ref is host_globals
        assert (
            host_globals["_get_candidate_cooldown_scope"]
            is cooldown_apply._get_candidate_cooldown_scope
        )
        assert host_globals["_state_manager"] is cooldown_apply._state_manager
        assert host_globals["status"] is status_sentinel
        assert host_globals["HTTPException"] is http_exception_sentinel
        assert host_globals["verbose_proxy_logger"] is logger_sentinel
        for name in cooldown_apply._HOST_FUNCTION_NAMES:
            assert host_globals[name] is getattr(cooldown_apply, name)
    finally:
        cooldown_apply._host_globals_ref = previous_host_globals


# ---------------------------------------------------------------------------
# _resolve_auto_agent_cooldown_publication_plan
# ---------------------------------------------------------------------------


class TestResolvePublicationPlan:
    def test_none_scope(self, configured_runtime: dict) -> None:
        configured_runtime["scope_fn"].return_value = "none"
        plan = _resolve_auto_agent_cooldown_publication_plan(
            request=None,
            candidate={"provider": "p", "model": "m"},
            lane_key="lane",
            selected_cooldown_key="ck",
            cooldown_seconds=60.0,
            error_class="some_error",
        )
        assert plan.applied_scope == "none"
        assert plan.memory_keys == ()
        assert plan.durable_keys == ()
        assert plan.duration_seconds == 60.0
        assert plan.request_local_action is None

    def test_request_local_scope(self, configured_runtime: dict) -> None:
        configured_runtime["scope_fn"].return_value = "request_local"
        plan = _resolve_auto_agent_cooldown_publication_plan(
            request=None,
            candidate={"provider": "p", "model": "m"},
            lane_key="lane",
            selected_cooldown_key="ck",
            cooldown_seconds=30.0,
            error_class="safety_policy_denied",
        )
        assert plan.applied_scope == "request_local"
        assert plan.memory_keys == ()
        assert plan.durable_keys == ()
        assert plan.request_local_action == "request_local_cooldown"
        assert plan.duration_seconds == 30.0

    def test_candidate_scope(self, configured_runtime: dict) -> None:
        configured_runtime["scope_fn"].return_value = "candidate"
        plan = _resolve_auto_agent_cooldown_publication_plan(
            request=None,
            candidate={"provider": "p", "model": "m"},
            lane_key="lane",
            selected_cooldown_key="ck",
            cooldown_seconds=120.0,
            error_class="rate_limited",
        )
        assert plan.applied_scope == "candidate"
        assert plan.memory_keys == ("ck",)
        assert plan.durable_keys == ("ck",)
        assert plan.duration_seconds == 120.0

    def test_candidate_scope_with_grok_account_lane(
        self, configured_runtime: dict
    ) -> None:
        configured_runtime["scope_fn"].return_value = "candidate"
        configured_runtime["grok_lane_fn"].return_value = "grok:__account_quota__:lane"
        plan = _resolve_auto_agent_cooldown_publication_plan(
            request=None,
            candidate={"provider": "xai", "model": "grok"},
            lane_key="lane",
            selected_cooldown_key="ck",
            cooldown_seconds=90.0,
            error_class="rate_limited",
            grok_account_quota_exhausted=True,
        )
        assert plan.applied_scope == "candidate"
        assert plan.memory_keys == ("ck", "grok:__account_quota__:lane")
        assert plan.durable_keys == ("ck", "grok:__account_quota__:lane")
        assert plan.grok_account_quota_exhausted is True

    def test_candidate_scope_grok_lane_same_as_selected(
        self, configured_runtime: dict
    ) -> None:
        configured_runtime["scope_fn"].return_value = "candidate"
        configured_runtime["grok_lane_fn"].return_value = "ck"
        plan = _resolve_auto_agent_cooldown_publication_plan(
            request=None,
            candidate={"provider": "xai", "model": "grok"},
            lane_key="lane",
            selected_cooldown_key="ck",
            cooldown_seconds=90.0,
            error_class="rate_limited",
            grok_account_quota_exhausted=True,
        )
        # Same key should not be duplicated
        assert plan.memory_keys == ("ck",)

    def test_managed_account_scope(self, configured_runtime: dict) -> None:
        configured_runtime["scope_fn"].return_value = "managed_account"
        plan = _resolve_auto_agent_cooldown_publication_plan(
            request=None,
            candidate={"provider": "kimi", "model": "kimi-code"},
            lane_key="lane",
            selected_cooldown_key="ck",
            cooldown_seconds=300.0,
            error_class="kimi_code_managed_account",
            kimi_failure_metadata={"scope": "managed_account"},
        )
        assert plan.applied_scope == "managed_account"
        assert plan.memory_keys == ("kimi:__managed__:default",)
        assert plan.durable_keys == ("kimi:__managed__:default",)
        assert plan.kimi_failure_metadata == {"scope": "managed_account"}

    def test_basic_pilot_gate_should_cool(self, configured_runtime: dict) -> None:
        configured_runtime["gate"]._decision = _FakeDecision(
            should_cool=True,
            duration_seconds=45.0,
            scope="candidate",
        )
        plan = _resolve_auto_agent_cooldown_publication_plan(
            request=None,
            candidate={},
            lane_key=None,
            selected_cooldown_key="rp:key",
            cooldown_seconds=0.0,
            error_class=None,
            is_basic_pilot_lane=True,
        )
        assert plan.applied_scope == "candidate"
        assert plan.memory_keys == ("rp:key",)
        assert plan.durable_keys == ("rp:key",)
        assert plan.duration_seconds == 45.0

    def test_basic_pilot_gate_should_not_cool(self, configured_runtime: dict) -> None:
        configured_runtime["gate"]._decision = _FakeDecision(should_cool=False)
        plan = _resolve_auto_agent_cooldown_publication_plan(
            request=None,
            candidate={},
            lane_key=None,
            selected_cooldown_key="rp:key",
            cooldown_seconds=0.0,
            error_class=None,
            is_basic_pilot_lane=True,
        )
        assert plan.applied_scope == "none"
        assert plan.memory_keys == ()
        assert plan.durable_keys == ()
        assert plan.duration_seconds == 0.0

    def test_basic_pilot_gate_scope_fallback(self, configured_runtime: dict) -> None:
        configured_runtime["gate"]._decision = _FakeDecision(
            should_cool=True,
            duration_seconds=10.0,
            scope=None,
        )
        plan = _resolve_auto_agent_cooldown_publication_plan(
            request=None,
            candidate={},
            lane_key=None,
            selected_cooldown_key="rp:key",
            cooldown_seconds=0.0,
            error_class=None,
            is_basic_pilot_lane=True,
        )
        assert plan.applied_scope == "candidate"

    def test_negative_cooldown_seconds_clamped(self, configured_runtime: dict) -> None:
        configured_runtime["scope_fn"].return_value = "candidate"
        plan = _resolve_auto_agent_cooldown_publication_plan(
            request=None,
            candidate={"provider": "p", "model": "m"},
            lane_key="lane",
            selected_cooldown_key="ck",
            cooldown_seconds=-5.0,
            error_class="rate_limited",
        )
        assert plan.duration_seconds == 0.0


# ---------------------------------------------------------------------------
# _persist_codex_cooldown_durable / _persist_anthropic_cooldown_durable
# ---------------------------------------------------------------------------


class TestPersistDurable:
    @pytest.mark.asyncio
    async def test_persist_codex_writes_each_key(
        self, configured_runtime: dict
    ) -> None:
        write = configured_runtime["write_durable"]
        await _persist_codex_cooldown_durable(keys=("k1", "k2"), seconds=60.0)
        assert write.await_count == 2
        calls = write.await_args_list
        assert calls[0].kwargs["alias_family"] == "codex"
        assert calls[0].kwargs["state_key"] == "k1"
        assert calls[1].kwargs["state_key"] == "k2"
        assert calls[0].kwargs["ttl_seconds"] == 60.0

    @pytest.mark.asyncio
    async def test_persist_codex_zero_seconds_skips(
        self, configured_runtime: dict
    ) -> None:
        write = configured_runtime["write_durable"]
        await _persist_codex_cooldown_durable(keys=("k1",), seconds=0.0)
        write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_persist_anthropic_writes_each_key(
        self, configured_runtime: dict
    ) -> None:
        write = configured_runtime["write_durable"]
        await _persist_anthropic_cooldown_durable(keys=("a1",), seconds=30.0)
        assert write.await_count == 1
        assert write.await_args_list[0].kwargs["alias_family"] == "anthropic"
        assert write.await_args_list[0].kwargs["state_key"] == "a1"

    @pytest.mark.asyncio
    async def test_persist_anthropic_negative_seconds_skips(
        self, configured_runtime: dict
    ) -> None:
        write = configured_runtime["write_durable"]
        await _persist_anthropic_cooldown_durable(keys=("a1",), seconds=-1.0)
        write.assert_not_awaited()


# ---------------------------------------------------------------------------
# _apply_auto_agent_alias_cooldown (shared)
# ---------------------------------------------------------------------------


class TestApplyAutoAgentAliasCooldown:
    @pytest.mark.asyncio
    async def test_none_scope_no_side_effects(
        self, configured_runtime: dict
    ) -> None:
        configured_runtime["scope_fn"].return_value = "none"
        req = _make_request()
        setter = AsyncMock()
        result = await _apply_auto_agent_alias_cooldown(
            request=req,
            candidate={"provider": "p", "model": "m"},
            lane_key="lane",
            selected_cooldown_key="ck",
            cooldown_seconds=60.0,
            error_class="err",
            set_candidate_cooldown=setter,
        )
        assert result == "none"
        setter.assert_not_awaited()
        configured_runtime["rl_set_fn"].assert_not_called()

    @pytest.mark.asyncio
    async def test_managed_account_scope_sets_managed_key(
        self, configured_runtime: dict
    ) -> None:
        configured_runtime["scope_fn"].return_value = "managed_account"
        req = _make_request()
        setter = AsyncMock()
        result = await _apply_auto_agent_alias_cooldown(
            request=req,
            candidate={"provider": "kimi", "model": "kimi-code"},
            lane_key="lane",
            selected_cooldown_key="ck",
            cooldown_seconds=300.0,
            error_class="kimi_code_managed_account",
            set_candidate_cooldown=setter,
        )
        assert result == "managed_account"
        setter.assert_awaited_once_with("kimi:__managed__:default", 300.0)

    @pytest.mark.asyncio
    async def test_candidate_scope_sets_selected_key(
        self, configured_runtime: dict
    ) -> None:
        configured_runtime["scope_fn"].return_value = "candidate"
        req = _make_request()
        setter = AsyncMock()
        result = await _apply_auto_agent_alias_cooldown(
            request=req,
            candidate={"provider": "p", "model": "m"},
            lane_key="lane",
            selected_cooldown_key="ck",
            cooldown_seconds=120.0,
            error_class="rate_limited",
            set_candidate_cooldown=setter,
        )
        assert result == "candidate"
        setter.assert_awaited_once_with("ck", 120.0)

    @pytest.mark.asyncio
    async def test_candidate_scope_grok_quota_adds_lane_key(
        self, configured_runtime: dict
    ) -> None:
        configured_runtime["scope_fn"].return_value = "candidate"
        configured_runtime["grok_lane_fn"].return_value = "grok:__account_quota__:lane"
        req = _make_request()
        setter = AsyncMock()
        result = await _apply_auto_agent_alias_cooldown(
            request=req,
            candidate={"provider": "xai", "model": "grok"},
            lane_key="lane",
            selected_cooldown_key="ck",
            cooldown_seconds=90.0,
            error_class="rate_limited",
            set_candidate_cooldown=setter,
            grok_account_quota_exhausted=True,
        )
        assert result == "candidate"
        assert setter.await_count == 2
        setter.assert_any_await("ck", 90.0)
        setter.assert_any_await("grok:__account_quota__:lane", 90.0)

    @pytest.mark.asyncio
    async def test_request_local_scope_sets_request_local(
        self, configured_runtime: dict
    ) -> None:
        configured_runtime["scope_fn"].return_value = "request_local"
        req = _make_request()
        setter = AsyncMock()
        result = await _apply_auto_agent_alias_cooldown(
            request=req,
            candidate={"provider": "p", "model": "m"},
            lane_key="lane",
            selected_cooldown_key="ck",
            cooldown_seconds=30.0,
            error_class="safety_policy_denied",
            set_candidate_cooldown=setter,
        )
        assert result == "request_local"
        setter.assert_not_awaited()
        configured_runtime["rl_key_fn"].assert_called_once_with(
            candidate={"provider": "p", "model": "m"},
            lane_key="lane",
        )
        configured_runtime["rl_set_fn"].assert_called_once_with(
            req,
            cooldown_key="rl:key",
            cooldown_seconds=30.0,
        )
        configured_runtime["rl_exclude_fn"].assert_called_once_with(
            req,
            cooldown_key="rl:key",
        )


# ---------------------------------------------------------------------------
# _apply_codex_auto_agent_alias_cooldown (Codex wrapper)
# ---------------------------------------------------------------------------


class TestApplyCodexWrapper:
    @pytest.mark.asyncio
    async def test_delegates_to_shared_apply(self, configured_runtime: dict) -> None:
        configured_runtime["scope_fn"].return_value = "candidate"
        req = _make_request()
        result = await _apply_codex_auto_agent_alias_cooldown(
            request=req,
            candidate={"provider": "p", "model": "m"},
            lane_key="lane",
            selected_cooldown_key="ck",
            cooldown_seconds=60.0,
            error_class="rate_limited",
        )
        assert result == "candidate"
        configured_runtime["codex_set"].assert_awaited_once_with("ck", 60.0)

    @pytest.mark.asyncio
    async def test_basic_pilot_lane_routes_to_gated(
        self, configured_runtime: dict
    ) -> None:
        configured_runtime["gate"]._decision = _FakeDecision(
            should_cool=True,
            duration_seconds=20.0,
            scope="candidate",
        )
        req = _make_request()
        with patch.object(
            cooldown_apply, "_apply_basic_pilot_gated_cooldown", new_callable=AsyncMock, return_value="candidate"
        ) as mock_gated:
            result = await _apply_codex_auto_agent_alias_cooldown(
                request=req,
                candidate={},
                lane_key=None,
                selected_cooldown_key="rp:key",
                cooldown_seconds=0.0,
                error_class=None,
                is_basic_pilot_lane=True,
            )
        assert result == "candidate"
        mock_gated.assert_awaited_once()


# ---------------------------------------------------------------------------
# _apply_anthropic_auto_agent_alias_cooldown (Anthropic wrapper)
# ---------------------------------------------------------------------------


class TestApplyAnthropicWrapper:
    @pytest.mark.asyncio
    async def test_delegates_to_shared_apply(self, configured_runtime: dict) -> None:
        configured_runtime["scope_fn"].return_value = "candidate"
        req = _make_request()
        result = await _apply_anthropic_auto_agent_alias_cooldown(
            request=req,
            candidate={"provider": "p", "model": "m"},
            lane_key="lane",
            selected_cooldown_key="ck",
            cooldown_seconds=60.0,
            error_class="rate_limited",
        )
        assert result == "candidate"
        configured_runtime["anthropic_set"].assert_awaited_once_with("ck", 60.0)

    @pytest.mark.asyncio
    async def test_ignores_basic_pilot_flag(self, configured_runtime: dict) -> None:
        configured_runtime["scope_fn"].return_value = "candidate"
        req = _make_request()
        result = await _apply_anthropic_auto_agent_alias_cooldown(
            request=req,
            candidate={"provider": "p", "model": "m"},
            lane_key="lane",
            selected_cooldown_key="ck",
            cooldown_seconds=60.0,
            error_class="rate_limited",
            is_basic_pilot_lane=True,
        )
        # Should still use shared apply, not basic-pilot gate
        assert result == "candidate"
        configured_runtime["anthropic_set"].assert_awaited_once()


# ---------------------------------------------------------------------------
# _apply_basic_pilot_gated_cooldown
# ---------------------------------------------------------------------------


class TestApplyBasicPilotGated:
    @pytest.mark.asyncio
    async def test_gate_cools_publishes_memory_and_durable(
        self, configured_runtime: dict
    ) -> None:
        configured_runtime["gate"]._decision = _FakeDecision(
            should_cool=True,
            duration_seconds=45.0,
            scope="candidate",
        )
        mgr = configured_runtime["mgr"]
        setter = AsyncMock()

        with patch.object(asyncio, "ensure_future") as mock_ensure:
            result = await _apply_basic_pilot_gated_cooldown(
                selected_cooldown_key="rp:key",
                set_candidate_cooldown=setter,
            )

        assert result == "candidate"
        # Memory publication happened synchronously
        remaining = mgr.codex.get_memory_cooldown_remaining("rp:key")
        assert remaining > 0
        # Durable write was fire-and-forget
        mock_ensure.assert_called_once()

    @pytest.mark.asyncio
    async def test_gate_not_cool_returns_none(
        self, configured_runtime: dict
    ) -> None:
        configured_runtime["gate"]._decision = _FakeDecision(should_cool=False)
        setter = AsyncMock()
        result = await _apply_basic_pilot_gated_cooldown(
            selected_cooldown_key="rp:key",
            set_candidate_cooldown=setter,
        )
        assert result == "none"
        setter.assert_not_awaited()


# ---------------------------------------------------------------------------
# _set_codex_auto_agent_candidate_cooldowns (compatibility entry point)
# ---------------------------------------------------------------------------


class TestSetCodexCandidateCooldowns:
    @pytest.mark.asyncio
    async def test_delegates_to_codex_wrapper(
        self, configured_runtime: dict
    ) -> None:
        configured_runtime["scope_fn"].return_value = "candidate"
        req = _make_request()
        result = await _set_codex_auto_agent_candidate_cooldowns(
            request=req,
            candidate={"provider": "p", "model": "m"},
            lane_key="lane",
            selected_cooldown_key="ck",
            cooldown_seconds=60.0,
            error_class="rate_limited",
        )
        assert result == "candidate"
        configured_runtime["codex_set"].assert_awaited_once_with("ck", 60.0)

    @pytest.mark.asyncio
    async def test_delegates_basic_pilot_flag(
        self, configured_runtime: dict
    ) -> None:
        configured_runtime["gate"]._decision = _FakeDecision(
            should_cool=True,
            duration_seconds=15.0,
            scope="candidate",
        )
        req = _make_request()
        with patch.object(
            cooldown_apply, "_apply_basic_pilot_gated_cooldown", new_callable=AsyncMock, return_value="candidate"
        ) as mock_gated:
            result = await _set_codex_auto_agent_candidate_cooldowns(
                request=req,
                candidate={},
                lane_key=None,
                selected_cooldown_key="rp:key",
                cooldown_seconds=0.0,
                error_class=None,
                is_basic_pilot_lane=True,
            )
        assert result == "candidate"
        mock_gated.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_request_local_scope_via_compat_entry(
        self, configured_runtime: dict
    ) -> None:
        configured_runtime["scope_fn"].return_value = "request_local"
        req = _make_request()
        result = await _set_codex_auto_agent_candidate_cooldowns(
            request=req,
            candidate={"provider": "p", "model": "m"},
            lane_key="lane",
            selected_cooldown_key="ck",
            cooldown_seconds=30.0,
            error_class="safety_policy_denied",
        )
        assert result == "request_local"
        configured_runtime["rl_set_fn"].assert_called_once()
        configured_runtime["rl_exclude_fn"].assert_called_once()


# ---------------------------------------------------------------------------
# Plan is CooldownPublicationPlan from interfaces.py
# ---------------------------------------------------------------------------


class TestPlanType:
    def test_resolver_returns_interfaces_plan(
        self, configured_runtime: dict
    ) -> None:
        configured_runtime["scope_fn"].return_value = "none"
        plan = _resolve_auto_agent_cooldown_publication_plan(
            request=None,
            candidate={},
            lane_key=None,
            selected_cooldown_key="ck",
            cooldown_seconds=0.0,
            error_class=None,
        )
        assert isinstance(plan, CooldownPublicationPlan)

    def test_plan_is_frozen(self, configured_runtime: dict) -> None:
        configured_runtime["scope_fn"].return_value = "candidate"
        plan = _resolve_auto_agent_cooldown_publication_plan(
            request=None,
            candidate={},
            lane_key=None,
            selected_cooldown_key="ck",
            cooldown_seconds=10.0,
            error_class="rate_limited",
        )
        with pytest.raises(AttributeError):
            plan.applied_scope = "mutated"  # type: ignore[misc]
