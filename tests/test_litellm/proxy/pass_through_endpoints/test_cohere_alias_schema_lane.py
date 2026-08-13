"""COHERE-001: Codex/direct-Cohere schema and lane selection.

Verifies the reusable ``cohere`` provider identity, the Codex-only adapter
route family, the credential-scoped Cohere lane key, and that the Codex
candidate preserves the existing candidate/cooldown key shape. Direct Cohere
state keys must stay distinct from OpenRouter state keys.

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
    REGISTERED_PROVIDERS,
    REGISTERED_ROUTE_FAMILIES,
)

_CODEX_COHERE_ROUTE_FAMILY = "codex_cohere_chat_completions_adapter"
_COHERE_MODEL = "cohere/command-a-03-2025"

_COHERE_ALIAS_YAML = """\
defaults: {}
aliases:
  - name: cohere-lane
    candidates:
      - provider: cohere
        model: cohere/command-a-03-2025
        route_family: codex_cohere_chat_completions_adapter
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


def _cohere_candidate(*, route_family: str) -> dict[str, Any]:
    return {
        "provider": policy.CODEX_AUTO_AGENT_COHERE_PROVIDER,
        "model": _COHERE_MODEL,
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


# ---------------------------------------------------------------------------
# Provider identity and route-family schema registration
# ---------------------------------------------------------------------------


class TestCohereRegistration:
    def test_cohere_provider_identity_registered(self):
        assert policy.CODEX_AUTO_AGENT_COHERE_PROVIDER == "cohere"
        assert policy.CODEX_AUTO_AGENT_COHERE_PROVIDER in REGISTERED_PROVIDERS

    def test_cohere_codex_route_family_is_codex_only(self):
        assert _CODEX_COHERE_ROUTE_FAMILY in REGISTERED_ROUTE_FAMILIES
        assert _CODEX_COHERE_ROUTE_FAMILY in CODEX_ONLY_ROUTE_FAMILIES

    def test_cohere_adapter_descriptors_are_cohere_native(self):
        config = adapter_config.CODEX_COHERE_CHAT_COMPLETIONS
        assert config.route_family in REGISTERED_ROUTE_FAMILIES
        assert config.credential_family == "cohere"
        assert config.expected_target_family == "cohere"
        assert config.custom_llm_provider == "cohere"
        assert config.adapter == _CODEX_COHERE_ROUTE_FAMILY


# ---------------------------------------------------------------------------
# Config compile: Codex admission, no ingress projection, fail-closed validation
# ---------------------------------------------------------------------------


class TestCohereCompile:
    def test_cohere_alias_compiles_without_ingress_projection(self):
        snapshot = compile_yaml(_COHERE_ALIAS_YAML)
        candidate = snapshot.aliases["cohere-lane"].candidates[0]
        assert candidate.provider == "cohere"
        assert candidate.model == _COHERE_MODEL
        assert candidate.route_family == _CODEX_COHERE_ROUTE_FAMILY
        assert candidate.anthropic_route_family is None

    def test_cohere_model_admission_is_config_driven(self):
        raw = """\
defaults: {}
aliases:
  - name: cohere-admission
    candidates:
      - provider: cohere
        model: cohere/any-future-model-id
        route_family: codex_cohere_chat_completions_adapter
        priority: 50
"""
        snapshot = compile_yaml(raw)
        assert "cohere-admission" in snapshot.aliases

    def test_cohere_alias_rejects_cross_ingress_override(self):
        raw = """\
defaults: {}
aliases:
  - name: cohere-cross-ingress
    candidates:
      - provider: cohere
        model: cohere/command-a-03-2025
        route_family: codex_cohere_chat_completions_adapter
        anthropic_route_family: codex_cohere_chat_completions_adapter
        priority: 50
"""
        with pytest.raises(ConfigCompileError):
            compile_yaml(raw)

    def test_unregistered_provider_still_fails_closed(self):
        raw = """\
defaults: {}
aliases:
  - name: cohere-typo
    candidates:
      - provider: cohere_typo
        model: cohere/command-a-03-2025
        route_family: codex_cohere_chat_completions_adapter
        priority: 50
"""
        with pytest.raises((ValidationError, ConfigCompileError)):
            compile_yaml(raw)

    def test_unregistered_route_family_still_fails_closed(self):
        raw = """\
defaults: {}
aliases:
  - name: cohere-bad-family
    candidates:
      - provider: cohere
        model: cohere/command-a-03-2025
        route_family: codex_cohere_responses_adapter
        priority: 50
"""
        with pytest.raises((ValidationError, ConfigCompileError)):
            compile_yaml(raw)


# ---------------------------------------------------------------------------
# Lane selection: credential-scoped key, both ingresses, OpenRouter isolation
# ---------------------------------------------------------------------------


class TestCohereLaneSelection:
    def test_lane_key_is_static_and_separate_from_openrouter(self):
        lane_key = policy.CODEX_AUTO_AGENT_COHERE_LANE_KEY
        assert lane_key == "cohere_native"
        assert lane_key != policy.CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY
        assert ":" not in lane_key

    @pytest.mark.asyncio
    async def test_codex_ingress_selects_cohere_lane(self, _selection_runtime):
        state = await selection._build_codex_auto_agent_candidate_state(
            _request("/v1/responses"),
            candidate_template=_cohere_candidate(
                route_family=_CODEX_COHERE_ROUTE_FAMILY
            ),
        )
        assert state["lane_key"] == policy.CODEX_AUTO_AGENT_COHERE_LANE_KEY
        assert state["cooldown_key"] == (
            f"cohere:{_COHERE_MODEL}:{policy.CODEX_AUTO_AGENT_COHERE_LANE_KEY}"
        )
        assert "skip_reason" not in state

    @pytest.mark.asyncio
    async def test_lane_key_does_not_depend_on_request_credentials(
        self, _selection_runtime
    ):
        request_a = _request(
            "/v1/responses",
            headers=[(b"authorization", b"Bearer cohere-key-alpha")],
        )
        request_b = _request(
            "/v1/responses",
            headers=[(b"authorization", b"Bearer cohere-key-beta")],
        )
        state_a = await selection._build_codex_auto_agent_candidate_state(
            request_a,
            candidate_template=_cohere_candidate(
                route_family=_CODEX_COHERE_ROUTE_FAMILY
            ),
        )
        state_b = await selection._build_codex_auto_agent_candidate_state(
            request_b,
            candidate_template=_cohere_candidate(
                route_family=_CODEX_COHERE_ROUTE_FAMILY
            ),
        )
        assert state_a["lane_key"] == state_b["lane_key"]
        assert state_a["cooldown_key"] == state_b["cooldown_key"]

    @pytest.mark.asyncio
    async def test_cohere_state_keys_are_isolated_from_openrouter(
        self, _selection_runtime
    ):
        request = _request("/v1/responses")
        cohere_state = await selection._build_codex_auto_agent_candidate_state(
            request,
            candidate_template=_cohere_candidate(
                route_family=_CODEX_COHERE_ROUTE_FAMILY
            ),
        )
        openrouter_state = await selection._build_codex_auto_agent_candidate_state(
            request,
            candidate_template={
                "provider": "openrouter",
                "model": "openrouter/some/paid-model",
                "route_family": "codex_openrouter_completion_adapter",
            },
        )
        assert cohere_state["lane_key"] != openrouter_state["lane_key"]
        assert (
            openrouter_state["lane_key"]
            == policy.CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY
        )
        assert cohere_state["cooldown_key"] != openrouter_state["cooldown_key"]

    @pytest.mark.asyncio
    async def test_candidate_shape_preserves_affinity_fields(
        self, _selection_runtime
    ):
        state = await selection._build_codex_auto_agent_candidate_state(
            _request("/v1/responses"),
            candidate_template=_cohere_candidate(
                route_family=_CODEX_COHERE_ROUTE_FAMILY
            ),
        )
        candidate = state["candidate"]
        assert candidate["provider"] == policy.CODEX_AUTO_AGENT_COHERE_PROVIDER
        assert candidate["model"] == _COHERE_MODEL
        assert candidate["route_family"] == _CODEX_COHERE_ROUTE_FAMILY
