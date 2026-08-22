"""NOUS-002: Codex/direct-Nous schema and lane selection.

Verifies the reusable ``nous`` provider identity, the Codex-only adapter
route family, the static Nous lane key, and that Wave B does not insert
Nous into production ``basic.yaml`` / ``work.yaml``.

No provider egress, no synthetic LLM calls, no live Hermes reads.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, patch

import pytest
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
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
    compile_directory,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_schema import (
    CODEX_ONLY_ROUTE_FAMILIES,
    CODEX_TO_ANTHROPIC_ROUTE_FAMILY,
    REGISTERED_PROVIDERS,
    REGISTERED_ROUTE_FAMILIES,
)

_CODEX_NOUS_ROUTE_FAMILY = "codex_nous_chat_completions_adapter"
_NOUS_MODEL = "stealth/ox-alpha"
_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
_BASIC_YAML_PATH = os.path.join(
    _REPO_ROOT, "litellm", "proxy", "aawm_alias_config", "basic.yaml"
)
_WORK_YAML_PATH = os.path.join(
    _REPO_ROOT, "litellm", "proxy", "aawm_alias_config", "work.yaml"
)

_NOUS_ALIAS_YAML = """\
defaults: {}
aliases:
  - name: nous-lane
    candidates:
      - provider: nous
        model: stealth/ox-alpha
        route_family: codex_nous_chat_completions_adapter
        priority: 50
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


def _nous_candidate(*, route_family: str, model: str = _NOUS_MODEL) -> dict[str, Any]:
    return {
        "provider": policy.CODEX_AUTO_AGENT_NOUS_PROVIDER,
        "model": model,
        "route_family": route_family,
    }


@pytest.fixture()
def _selection_runtime():
    """Bind selection seams with inert stubs; restore module state afterwards."""
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


class TestNousRegistration:
    def test_nous_provider_identity_registered(self):
        assert policy.CODEX_AUTO_AGENT_NOUS_PROVIDER == "nous"
        assert policy.CODEX_AUTO_AGENT_NOUS_PROVIDER in REGISTERED_PROVIDERS

    def test_nous_codex_route_family_is_codex_only(self):
        assert _CODEX_NOUS_ROUTE_FAMILY in REGISTERED_ROUTE_FAMILIES
        assert _CODEX_NOUS_ROUTE_FAMILY in CODEX_ONLY_ROUTE_FAMILIES
        assert _CODEX_NOUS_ROUTE_FAMILY not in CODEX_TO_ANTHROPIC_ROUTE_FAMILY
        assert "anthropic_nous_completion_adapter" not in REGISTERED_ROUTE_FAMILIES

        config = adapter_config.CODEX_NOUS_CHAT_COMPLETIONS
        assert config.route_family == _CODEX_NOUS_ROUTE_FAMILY
        assert config.credential_family == "nous"
        assert config.expected_target_family == "nous"
        assert config.custom_llm_provider == "nous"
        assert config.target_endpoint_label == "nous:/v1/chat/completions"


class TestNousCompile:
    def test_nous_alias_compiles_without_ingress_projection(self):
        snapshot = compile_yaml(_NOUS_ALIAS_YAML)
        candidate = snapshot.aliases["nous-lane"].candidates[0]
        assert candidate.provider == "nous"
        assert candidate.model == _NOUS_MODEL
        assert candidate.route_family == _CODEX_NOUS_ROUTE_FAMILY
        assert candidate.anthropic_route_family is None

    def test_nous_alias_rejects_cross_ingress_override(self):
        raw = """\
defaults: {}
aliases:
  - name: nous-cross-ingress
    candidates:
      - provider: nous
        model: stealth/ox-alpha
        route_family: codex_nous_chat_completions_adapter
        anthropic_route_family: anthropic_openrouter_completion_adapter
        priority: 50
"""
        with pytest.raises(ConfigCompileError):
            compile_yaml(raw)

    def test_wave_b_does_not_insert_nous_into_basic_or_work(self):
        with open(_BASIC_YAML_PATH, "r", encoding="utf-8") as handle:
            basic_snapshot = compile_yaml(handle.read())
        work_snapshot = compile_directory(
            Path(_REPO_ROOT) / "litellm" / "proxy" / "aawm_alias_config"
        )

        basic_providers = {
            candidate.provider
            for candidate in basic_snapshot.aliases["basic"].candidates
            if getattr(candidate, "provider", None)
        }
        work_providers = {
            candidate.provider
            for candidate in work_snapshot.aliases["work"].candidates
            if getattr(candidate, "provider", None)
        }
        assert "nous" not in basic_providers
        assert "nous" not in work_providers


class TestNousLaneSelection:
    def test_lane_key_is_static_and_separate_from_openrouter_and_openai(self):
        lane_key = policy.CODEX_AUTO_AGENT_NOUS_LANE_KEY
        assert lane_key == "nous"
        assert ":" not in lane_key
        assert lane_key != policy.CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY
        assert lane_key != getattr(policy, "CODEX_AUTO_AGENT_OPENCODE_GO_LANE_KEY")
        assert lane_key != getattr(policy, "CODEX_AUTO_AGENT_OPENCODE_LANE_KEY")
        assert lane_key != "openai:primary"
        assert lane_key != getattr(
            policy, "CODEX_AUTO_AGENT_OPENAI_LANE_KEY", "openai:primary"
        )

    @pytest.mark.asyncio
    async def test_codex_ingress_selects_nous_lane(self, _selection_runtime):
        state = await selection._build_codex_auto_agent_candidate_state(
            _request("/v1/responses"),
            candidate_template=_nous_candidate(route_family=_CODEX_NOUS_ROUTE_FAMILY),
        )
        assert state["lane_key"] == policy.CODEX_AUTO_AGENT_NOUS_LANE_KEY
        assert state["cooldown_key"] == (
            f"nous:{_NOUS_MODEL}:{policy.CODEX_AUTO_AGENT_NOUS_LANE_KEY}"
        )
        assert "skip_reason" not in state

    @pytest.mark.asyncio
    async def test_nous_does_not_fall_through_to_openai_lane(self, _selection_runtime):
        state = await selection._build_codex_auto_agent_candidate_state(
            _request("/v1/responses"),
            candidate_template=_nous_candidate(route_family=_CODEX_NOUS_ROUTE_FAMILY),
        )
        openai_lane_key = "openai:primary"
        assert state["lane_key"] != openai_lane_key
        assert state["lane_key"] == "nous"
        assert not str(state["cooldown_key"]).startswith("openai:")
        assert "openai:primary" not in str(state["cooldown_key"])
