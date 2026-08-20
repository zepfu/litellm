"""Z.AI Coding Plan Codex-only schema and lane selection.

Clones the Cohere Codex-only compile path. Coding Plan must register as a
Codex-only route family with no Anthropic projection, and Codex lane keys
must stay distinct from Alibaba Token Plan and OpenRouter.

No provider egress, no synthetic LLM calls.
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError
from starlette.requests import Request

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    adapter_config,
    policy,
    selection,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_compiler import (
    ConfigCompileError,
    compile_yaml,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_schema import (
    CODEX_ONLY_ROUTE_FAMILIES,
    CODEX_TO_ANTHROPIC_ROUTE_FAMILY,
    REGISTERED_PROVIDERS,
    REGISTERED_ROUTE_FAMILIES,
)

_CODEX_ZAI_ROUTE_FAMILY = "codex_zai_coding_plan_chat_completions_adapter"
_ZAI_CODING_PLAN_MODEL = "zai_coding_plan/glm-5.3"

_ZAI_CODING_PLAN_ALIAS_YAML = """\
defaults: {}
aliases:
  - name: coding-plan-lane
    candidates:
      - provider: zai_coding_plan
        model: zai_coding_plan/glm-5.3
        route_family: codex_zai_coding_plan_chat_completions_adapter
        priority: 110
"""


def _request(path: str, *, headers: list[tuple[bytes, bytes]] | None = None) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": headers or [],
            "client": ("127.0.0.1", 43123),
            "server": ("testserver", 80),
        }
    )


def _coding_plan_candidate(*, route_family: str) -> dict[str, Any]:
    return {
        "provider": policy.CODEX_AUTO_AGENT_ZAI_CODING_PLAN_PROVIDER,
        "model": _ZAI_CODING_PLAN_MODEL,
        "route_family": route_family,
    }


@pytest.fixture()
def _selection_runtime():
    runtime_names = {
        "_get_codex_active_cooldown_state",
        "_get_anthropic_active_cooldown_state",
        "_get_anthropic_merged_codex_openai_cooldown_state",
        "_set_codex_cooldown",
        "_set_anthropic_cooldown",
        "_get_codex_session_affinity",
        "_get_anthropic_session_affinity",
        "_get_openrouter_adapter_active_cooldown_seconds",
        "_extract_client_product_label",
        "_resolve_codex_session_key",
        "_resolve_anthropic_session_key",
        "_has_continuation_state",
        "_is_grok_account_quota_candidate",
        "_get_grok_account_quota_lane_cooldown_key",
        "_is_kimi_code_candidate",
        "_get_kimi_managed_account_cooldown_key",
        "_get_codex_quota_observation_pool",
        "_get_codex_quota_observation_environment",
    }
    previous_runtime = {name: getattr(selection, name) for name in runtime_names}

    async def _zero_cooldown_state(key: str) -> tuple[float, str]:
        return (0.0, "local_fallback")

    async def _noop_cooldown(key: str, seconds: float) -> None:
        pass

    async def _zero_adapter(model: Optional[str]) -> float:
        return 0.0

    selection.configure_selection_runtime(
        get_codex_active_cooldown_state=_zero_cooldown_state,
        get_anthropic_active_cooldown_state=_zero_cooldown_state,
        get_anthropic_merged_codex_openai_cooldown_state=_zero_cooldown_state,
        set_codex_cooldown=_noop_cooldown,
        set_anthropic_cooldown=_noop_cooldown,
        get_codex_session_affinity=AsyncMock(return_value=None),
        get_anthropic_session_affinity=AsyncMock(return_value=None),
        get_openrouter_adapter_active_cooldown_seconds=_zero_adapter,
        extract_client_product_label=lambda r, b: None,
        resolve_codex_session_key=lambda r, b, *, alias_model: None,
        resolve_anthropic_session_key=lambda r, b, *, alias_model: None,
        has_continuation_state=lambda v: False,
        is_grok_account_quota_candidate=lambda c: False,
        get_grok_account_quota_lane_cooldown_key=lambda c, lk: None,
        is_kimi_code_candidate=lambda c: (
            isinstance(c, dict) and c.get("provider") == "kimi_code"
        ),
        get_kimi_managed_account_cooldown_key=(
            lambda: "kimi_code:__managed_account__:kimi_code_managed_account"
        ),
    )
    runtime_globals = selection._build_codex_auto_agent_candidate_state.__globals__
    overrides = {
        "_resolve_codex_auto_agent_openai_cooldown_lane_key": (
            lambda request: "openai:primary"
        ),
        "_resolve_anthropic_auto_agent_native_cooldown_lane_key": (
            lambda request: "anthropic:primary"
        ),
    }
    try:
        with patch.dict(runtime_globals, overrides):
            yield
    finally:
        for name, value in previous_runtime.items():
            setattr(selection, name, value)


class TestZaiCodingPlanRegistration:
    def test_coding_plan_provider_identity_registered(self):
        assert policy.CODEX_AUTO_AGENT_ZAI_CODING_PLAN_PROVIDER == "zai_coding_plan"
        assert policy.CODEX_AUTO_AGENT_ZAI_CODING_PLAN_PROVIDER in REGISTERED_PROVIDERS

    def test_coding_plan_codex_route_family_is_codex_only(self):
        assert _CODEX_ZAI_ROUTE_FAMILY in REGISTERED_ROUTE_FAMILIES
        assert _CODEX_ZAI_ROUTE_FAMILY in CODEX_ONLY_ROUTE_FAMILIES
        assert _CODEX_ZAI_ROUTE_FAMILY not in CODEX_TO_ANTHROPIC_ROUTE_FAMILY

    def test_coding_plan_adapter_descriptors_are_coding_plan_native(self):
        config = adapter_config.CODEX_ZAI_CODING_PLAN
        assert config.route_family in REGISTERED_ROUTE_FAMILIES
        assert config.credential_family == "zai_coding_plan"
        assert config.expected_target_family == "zai_coding_plan"
        assert config.custom_llm_provider == "zai_coding_plan"
        assert config.adapter == _CODEX_ZAI_ROUTE_FAMILY
        assert not hasattr(adapter_config, "ANTHROPIC_ZAI_CODING_PLAN")


class TestZaiCodingPlanCompile:
    def test_coding_plan_alias_compiles_without_ingress_projection(self):
        snapshot = compile_yaml(_ZAI_CODING_PLAN_ALIAS_YAML)
        candidate = snapshot.aliases["coding-plan-lane"].candidates[0]
        assert candidate.provider == "zai_coding_plan"
        assert candidate.model == _ZAI_CODING_PLAN_MODEL
        assert candidate.route_family == _CODEX_ZAI_ROUTE_FAMILY
        assert candidate.anthropic_route_family is None

    def test_coding_plan_alias_rejects_cross_ingress_override(self):
        raw = """\
defaults: {}
aliases:
  - name: coding-plan-cross-ingress
    candidates:
      - provider: zai_coding_plan
        model: zai_coding_plan/glm-5.3
        route_family: codex_zai_coding_plan_chat_completions_adapter
        anthropic_route_family: anthropic_alibaba_token_plan_chat_completions_adapter
        priority: 110
"""
        with pytest.raises(ConfigCompileError):
            compile_yaml(raw)

    def test_unregistered_provider_still_fails_closed(self):
        raw = """\
defaults: {}
aliases:
  - name: coding-plan-typo
    candidates:
      - provider: zai_coding_plan_typo
        model: zai_coding_plan/glm-5.3
        route_family: codex_zai_coding_plan_chat_completions_adapter
        priority: 110
"""
        with pytest.raises((ValidationError, ConfigCompileError)):
            compile_yaml(raw)

    def test_unregistered_route_family_still_fails_closed(self):
        raw = """\
defaults: {}
aliases:
  - name: coding-plan-bad-family
    candidates:
      - provider: zai_coding_plan
        model: zai_coding_plan/glm-5.3
        route_family: anthropic_zai_coding_plan_chat_completions_adapter
        priority: 110
"""
        with pytest.raises((ValidationError, ConfigCompileError)):
            compile_yaml(raw)

    def test_coding_plan_provider_rejects_foreign_route_family(self):
        raw = """\
defaults: {}
aliases:
  - name: coding-plan-wrong-family
    candidates:
      - provider: zai_coding_plan
        model: zai_coding_plan/glm-5.3
        route_family: codex_alibaba_token_plan_chat_completions_adapter
        priority: 110
"""
        with pytest.raises(ConfigCompileError):
            compile_yaml(raw)


class TestZaiCodingPlanLaneSelection:
    def test_lane_key_is_static_and_separate_from_alibaba_and_openrouter(self):
        lane_key = policy.CODEX_AUTO_AGENT_ZAI_CODING_PLAN_LANE_KEY
        assert lane_key == "zai_coding_plan"
        assert lane_key != policy.CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY
        assert lane_key != policy.CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY
        assert ":" not in lane_key

    @pytest.mark.asyncio
    async def test_codex_ingress_selects_coding_plan_lane(self, _selection_runtime):
        state = await selection._build_codex_auto_agent_candidate_state(
            _request("/v1/responses"),
            candidate_template=_coding_plan_candidate(
                route_family=_CODEX_ZAI_ROUTE_FAMILY
            ),
        )
        assert state["lane_key"] == policy.CODEX_AUTO_AGENT_ZAI_CODING_PLAN_LANE_KEY
        assert state["cooldown_key"] == (
            "zai_coding_plan:"
            f"{_ZAI_CODING_PLAN_MODEL}:"
            f"{policy.CODEX_AUTO_AGENT_ZAI_CODING_PLAN_LANE_KEY}"
        )
        assert "skip_reason" not in state

    @pytest.mark.asyncio
    async def test_initialized_proxy_facade_resolves_coding_plan_lane(
        self, _selection_runtime
    ):
        from litellm.proxy.pass_through_endpoints import (
            llm_passthrough_endpoints as lpe,
        )

        state = await lpe._build_codex_auto_agent_candidate_state(
            _request("/v1/responses"),
            candidate_template=_coding_plan_candidate(
                route_family=_CODEX_ZAI_ROUTE_FAMILY
            ),
        )

        assert state["lane_key"] == policy.CODEX_AUTO_AGENT_ZAI_CODING_PLAN_LANE_KEY
        assert "skip_reason" not in state

    @pytest.mark.asyncio
    async def test_coding_plan_state_keys_are_isolated_from_alibaba(
        self, _selection_runtime
    ):
        request = _request("/v1/responses")
        coding_plan_state = await selection._build_codex_auto_agent_candidate_state(
            request,
            candidate_template=_coding_plan_candidate(
                route_family=_CODEX_ZAI_ROUTE_FAMILY
            ),
        )
        alibaba_state = await selection._build_codex_auto_agent_candidate_state(
            request,
            candidate_template={
                "provider": "alibaba_token_plan",
                "model": "alibaba_token_plan/glm-5.2",
                "route_family": "codex_alibaba_token_plan_chat_completions_adapter",
            },
        )
        assert coding_plan_state["lane_key"] != alibaba_state["lane_key"]
        assert coding_plan_state["cooldown_key"] != alibaba_state["cooldown_key"]
