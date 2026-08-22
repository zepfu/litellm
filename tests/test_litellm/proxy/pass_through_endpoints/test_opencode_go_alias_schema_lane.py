"""OC-001 / OR-001: OpenCode Go Codex-only schema and lane selection.

Verifies the reusable ``opencode_go`` provider identity, the Codex-only Go
adapter route family, the static Go cooldown lane, and that compiled ``basic``
/ ``work`` YAML place Go then OpenRouter ox-alpha first.

No provider egress, no synthetic LLM calls.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, patch

import pytest
from starlette.requests import Request

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import policy, selection
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
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
    compile_directory,
)


_CODEX_GO_ROUTE_FAMILY = "codex_opencode_go_adapter"
_GO_MODEL = "ox-alpha-free"
_OPENROUTER_OX_ALPHA_MODEL = "openrouter/stealth/ox-alpha"
_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
_BASIC_YAML_PATH = os.path.join(
    _REPO_ROOT, "litellm", "proxy", "aawm_alias_config", "basic.yaml"
)
_ALIAS_CONFIG_DIR = Path(_REPO_ROOT) / "litellm" / "proxy" / "aawm_alias_config"
_CANONICAL_COST_MAP_PATH = Path(_REPO_ROOT) / "model_prices_and_context_window.json"
_BUNDLED_COST_MAP_PATH = (
    Path(_REPO_ROOT) / "litellm" / "bundled_model_prices_and_context_window_fallback.json"
)

_GO_ALIAS_YAML = """\
defaults: {}
aliases:
  - name: go-lane
    candidates:
      - provider: opencode_go
        model: ox-alpha-free
        route_family: codex_opencode_go_adapter
        priority: 100
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


def _go_candidate(*, route_family: str) -> dict[str, Any]:
    return {
        "provider": policy.OPENCODE_GO_PROVIDER,
        "model": _GO_MODEL,
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


class TestOpenCodeGoRegistration:
    def test_opencode_go_provider_identity_registered(self):
        assert policy.OPENCODE_GO_PROVIDER == "opencode_go"
        assert policy.OPENCODE_GO_PROVIDER in REGISTERED_PROVIDERS

    def test_opencode_go_codex_route_family_is_codex_only(self):
        assert _CODEX_GO_ROUTE_FAMILY in REGISTERED_ROUTE_FAMILIES
        assert _CODEX_GO_ROUTE_FAMILY in CODEX_ONLY_ROUTE_FAMILIES
        assert _CODEX_GO_ROUTE_FAMILY not in CODEX_TO_ANTHROPIC_ROUTE_FAMILY
        assert (
            "anthropic_opencode_go_completion_adapter"
            not in REGISTERED_ROUTE_FAMILIES
        )


class TestOpenCodeGoCompile:
    def test_go_alias_compiles_without_ingress_projection(self):
        snapshot = compile_yaml(_GO_ALIAS_YAML)
        candidate = snapshot.aliases["go-lane"].candidates[0]
        assert candidate.provider == "opencode_go"
        assert candidate.model == _GO_MODEL
        assert candidate.route_family == _CODEX_GO_ROUTE_FAMILY
        assert candidate.anthropic_route_family is None

    def test_go_alias_rejects_cross_ingress_override(self):
        raw = """\
defaults: {}
aliases:
  - name: go-cross-ingress
    candidates:
      - provider: opencode_go
        model: ox-alpha-free
        route_family: codex_opencode_go_adapter
        anthropic_route_family: anthropic_openrouter_completion_adapter
        priority: 100
"""
        with pytest.raises(ConfigCompileError):
            compile_yaml(raw)

    def test_basic_yaml_places_go_then_openrouter_ox_alpha_first(self):
        with open(_BASIC_YAML_PATH, "r", encoding="utf-8") as handle:
            snapshot = compile_yaml(handle.read())

        basic_candidates = snapshot.aliases["basic"].candidates
        first = basic_candidates[0]
        second = basic_candidates[1]
        third = basic_candidates[2]
        assert (
            first.provider,
            first.model,
            first.route_family,
        ) == (
            "opencode_go",
            _GO_MODEL,
            _CODEX_GO_ROUTE_FAMILY,
        )
        assert first.priority == 100
        assert first.anthropic_route_family is None
        assert (
            second.provider,
            second.model,
            second.route_family,
        ) == (
            "nous",
            "stealth/ox-alpha",
            "codex_nous_chat_completions_adapter",
        )
        assert second.priority == 97
        assert second.anthropic_route_family is None
        assert (
            third.provider,
            third.model,
            third.route_family,
        ) == (
            "openrouter",
            _OPENROUTER_OX_ALPHA_MODEL,
            "codex_openrouter_completion_adapter",
        )
        assert third.priority == 95

        north_pairs = [
            (candidate.provider, candidate.model, candidate.route_family)
            for candidate in basic_candidates
            if candidate.model
            in {
                "cohere/north-mini-code-1-0",
                "openrouter/cohere/north-mini-code:free",
            }
        ]
        assert north_pairs == [
            (
                "cohere",
                "cohere/north-mini-code-1-0",
                "codex_cohere_chat_completions_adapter",
            ),
            (
                "openrouter",
                "openrouter/cohere/north-mini-code:free",
                "codex_openrouter_completion_adapter",
            ),
        ]
        cohere_index = next(
            index
            for index, candidate in enumerate(basic_candidates)
            if candidate.model == "cohere/north-mini-code-1-0"
        )
        assert cohere_index == 3
        assert basic_candidates[cohere_index - 1].model == _OPENROUTER_OX_ALPHA_MODEL

    def test_work_yaml_places_go_then_openrouter_ox_alpha_first(self):
        snapshot = compile_directory(_ALIAS_CONFIG_DIR)
        work_candidates = snapshot.aliases["work"].candidates
        first = work_candidates[0]
        second = work_candidates[1]
        third = work_candidates[2]
        fourth = work_candidates[3]
        assert (
            first.provider,
            first.model,
            first.route_family,
        ) == (
            "opencode_go",
            _GO_MODEL,
            _CODEX_GO_ROUTE_FAMILY,
        )
        assert first.priority == 110
        assert first.anthropic_route_family is None
        assert (
            second.provider,
            second.model,
            second.route_family,
        ) == (
            "nous",
            "stealth/ox-alpha",
            "codex_nous_chat_completions_adapter",
        )
        assert second.priority == 107
        assert second.anthropic_route_family is None
        assert (
            third.provider,
            third.model,
            third.route_family,
        ) == (
            "openrouter",
            _OPENROUTER_OX_ALPHA_MODEL,
            "codex_openrouter_completion_adapter",
        )
        assert third.priority == 105
        assert fourth.model == "gpt-5.3-codex-spark"


class TestOpenCodeGoLaneSelection:
    def test_lane_key_is_static_and_separate_from_zen_and_openrouter(self):
        lane_key = policy.CODEX_AUTO_AGENT_OPENCODE_GO_LANE_KEY
        assert lane_key == "opencode_go"
        assert lane_key != policy.CODEX_AUTO_AGENT_OPENCODE_LANE_KEY
        assert lane_key != policy.CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY
        assert ":" not in lane_key

    @pytest.mark.asyncio
    async def test_codex_ingress_selects_go_lane(self, _selection_runtime):
        state = await selection._build_codex_auto_agent_candidate_state(
            _request("/v1/responses"),
            candidate_template=_go_candidate(route_family=_CODEX_GO_ROUTE_FAMILY),
        )
        assert state["lane_key"] == policy.CODEX_AUTO_AGENT_OPENCODE_GO_LANE_KEY
        assert state["cooldown_key"] == (
            f"opencode_go:{_GO_MODEL}:{policy.CODEX_AUTO_AGENT_OPENCODE_GO_LANE_KEY}"
        )
        assert "skip_reason" not in state


class TestOpenCodeGoCostMap:
    def test_cost_map_contains_opencode_go_and_openrouter_ox_alpha_rows(self):
        canonical = json.loads(_CANONICAL_COST_MAP_PATH.read_text(encoding="utf-8"))
        bundled = json.loads(_BUNDLED_COST_MAP_PATH.read_text(encoding="utf-8"))

        go_row = canonical["opencode/ox-alpha-free"]
        assert go_row["litellm_provider"] == "opencode_go"
        assert go_row["input_cost_per_token"] == 0
        assert go_row["output_cost_per_token"] == 0
        assert go_row.get("cache_read_input_token_cost", 0) == 0
        assert go_row.get("cache_creation_input_token_cost", 0) == 0

        openrouter_row = canonical["openrouter/stealth/ox-alpha"]
        assert openrouter_row["litellm_provider"] == "openrouter"
        assert openrouter_row.get("input_cost_per_token", 0) == 0
        assert openrouter_row.get("output_cost_per_token", 0) == 0
        pricing = openrouter_row["provider_specific_entry"]["openrouter"][
            "aawm_reference_pricing"
        ]
        assert pricing["status"] == "unpriced"
        assert pricing["kind"] in {
            "hosted_alias_reference",
            "hosted_route_reference",
        }

        assert bundled["opencode/ox-alpha-free"] == go_row
        assert bundled["openrouter/stealth/ox-alpha"] == openrouter_row
