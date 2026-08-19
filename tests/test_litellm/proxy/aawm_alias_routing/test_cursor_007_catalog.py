"""CURSOR-007: Composer 2.5 and Cursor Grok catalog aliases.

Focused compile/snapshot and catalog-identity checks. No live Cursor
traffic and no CLI spawn.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from litellm.proxy._types import ProxyException
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.codex_candidate_calls import (
    _raise_cursor_agent_alias_not_implemented,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    adapter_config,
    policy,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_compiler import (
    ConfigCompileError,
    compile_yaml,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_schema import (
    CODEX_TO_ANTHROPIC_ROUTE_FAMILY,
    REGISTERED_PROVIDERS,
    REGISTERED_ROUTE_FAMILIES,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
    DEFAULT_CONFIG_DIR,
    compile_directory,
)
from litellm.types.utils import LlmProviders


_REPO_ROOT = Path(__file__).resolve().parents[4]
_BASIC_YAML_PATH = _REPO_ROOT / "litellm" / "proxy" / "aawm_alias_config" / "basic.yaml"
_SOTA_XAI_YAML_PATH = (
    _REPO_ROOT / "litellm" / "proxy" / "aawm_alias_config" / "sota-xai.yaml"
)
_CANONICAL_CATALOG = _REPO_ROOT / "model_prices_and_context_window.json"
_BUNDLED_CATALOG = (
    _REPO_ROOT / "litellm" / "bundled_model_prices_and_context_window_fallback.json"
)

_COMPOSER_MODEL = "cursor_agent/composer-2.5"
_CURSOR_GROK_MODEL = "cursor_agent/cursor-grok-4.6-high"
_CODEX_CURSOR_ROUTE_FAMILY = "codex_cursor_agent_aiserver_adapter"
_ANTHROPIC_CURSOR_ROUTE_FAMILY = "anthropic_cursor_agent_aiserver_adapter"


def _candidate_by_model(candidates, model: str):
    return next(candidate for candidate in candidates if candidate.model == model)


class TestCursorAgentRegistration:
    def test_cursor_agent_provider_is_registered_and_not_cloud_agents(self):
        assert policy.CODEX_AUTO_AGENT_CURSOR_AGENT_PROVIDER == "cursor_agent"
        assert policy.CODEX_AUTO_AGENT_CURSOR_AGENT_PROVIDER in REGISTERED_PROVIDERS
        assert "cursor" not in REGISTERED_PROVIDERS
        assert LlmProviders.CURSOR_AGENT.value == "cursor_agent"
        assert LlmProviders.CURSOR.value == "cursor"
        assert LlmProviders.CURSOR_AGENT.value != LlmProviders.CURSOR.value

    def test_cursor_agent_route_families_are_registered_and_mapped(self):
        assert _CODEX_CURSOR_ROUTE_FAMILY in REGISTERED_ROUTE_FAMILIES
        assert _ANTHROPIC_CURSOR_ROUTE_FAMILY in REGISTERED_ROUTE_FAMILIES
        assert (
            CODEX_TO_ANTHROPIC_ROUTE_FAMILY[_CODEX_CURSOR_ROUTE_FAMILY]
            == _ANTHROPIC_CURSOR_ROUTE_FAMILY
        )

    def test_cursor_agent_adapter_descriptors_stay_on_cursor_agent(self):
        codex = adapter_config.CODEX_CURSOR_AGENT_AISERVER
        anthropic = adapter_config.ANTHROPIC_CURSOR_AGENT_AISERVER
        assert codex.route_family == _CODEX_CURSOR_ROUTE_FAMILY
        assert anthropic.route_family == _ANTHROPIC_CURSOR_ROUTE_FAMILY
        assert codex.custom_llm_provider == "cursor_agent"
        assert anthropic.custom_llm_provider == "cursor_agent"
        assert codex.credential_family == "cursor_agent"
        assert anthropic.credential_family == "cursor_agent"
        assert "cursor.com" not in codex.target_endpoint_label
        assert "api.cursor.com" not in anthropic.target_endpoint_label


class TestCursorAgentCompile:
    def test_basic_yaml_places_composer_after_deepseek_flash_at_priority_42(self):
        snapshot = compile_yaml(_BASIC_YAML_PATH.read_text(encoding="utf-8"))
        candidates = snapshot.aliases["basic"].candidates
        models = [candidate.model for candidate in candidates]
        composer = _candidate_by_model(candidates, _COMPOSER_MODEL)
        deepseek = _candidate_by_model(
            candidates, "alibaba_token_plan/deepseek-v4-flash-0731"
        )
        qwen = _candidate_by_model(candidates, "alibaba_token_plan/qwen3.6-flash")

        assert composer.provider == "cursor_agent"
        assert composer.route_family == _CODEX_CURSOR_ROUTE_FAMILY
        assert composer.anthropic_route_family == _ANTHROPIC_CURSOR_ROUTE_FAMILY
        assert composer.priority == 42
        assert deepseek.priority == 45
        assert qwen.priority == 40
        assert models.index("alibaba_token_plan/deepseek-v4-flash-0731") < models.index(
            _COMPOSER_MODEL
        )
        assert models.index(_COMPOSER_MODEL) < models.index(
            "alibaba_token_plan/qwen3.6-flash"
        )
        assert "composer-2.5-fast" not in models
        assert all(candidate.provider != "cursor" for candidate in candidates)

    def test_sota_xai_prefers_managed_xai_over_cursor_grok(self):
        snapshot = compile_yaml(_SOTA_XAI_YAML_PATH.read_text(encoding="utf-8"))
        candidates = snapshot.aliases["sota-xai"].candidates
        xai = _candidate_by_model(candidates, "oa_xai/grok-4.6")
        cursor_grok = _candidate_by_model(candidates, _CURSOR_GROK_MODEL)

        assert [candidate.model for candidate in candidates] == [
            "oa_xai/grok-4.6",
            _CURSOR_GROK_MODEL,
        ]
        assert xai.provider == "xai"
        assert xai.priority == 100
        assert cursor_grok.provider == "cursor_agent"
        assert cursor_grok.priority == 90
        assert cursor_grok.route_family == _CODEX_CURSOR_ROUTE_FAMILY
        assert cursor_grok.anthropic_route_family == _ANTHROPIC_CURSOR_ROUTE_FAMILY
        assert xai.priority > cursor_grok.priority

    def test_directory_compile_inherits_cursor_grok_on_work_other(self):
        snapshot = compile_directory(DEFAULT_CONFIG_DIR)
        sota_xai = snapshot.aliases["sota-xai"].candidates
        assert [candidate.model for candidate in sota_xai] == [
            "oa_xai/grok-4.6",
            _CURSOR_GROK_MODEL,
        ]
        assert "work-other" in snapshot.aliases
        assert any(
            getattr(entry, "alias_name", None) == "sota-xai"
            for entry in snapshot.aliases["work-other"].candidates
        )

    def test_cursor_agent_rejects_cloud_agents_or_xai_route_family(self):
        raw = """\
defaults: {}
aliases:
  - name: cursor-bad-family
    candidates:
      - provider: cursor_agent
        model: cursor_agent/composer-2.5
        route_family: codex_xai_oauth_responses_adapter
        priority: 42
"""
        with pytest.raises(ConfigCompileError, match="Cursor Agent"):
            compile_yaml(raw)

    def test_unregistered_cursor_provider_still_fails_closed(self):
        raw = """\
defaults: {}
aliases:
  - name: cursor-typo
    candidates:
      - provider: cursor
        model: cursor_agent/composer-2.5
        route_family: codex_cursor_agent_aiserver_adapter
        priority: 42
"""
        with pytest.raises((ValidationError, ConfigCompileError)):
            compile_yaml(raw)


class TestCursorAgentCatalogIdentity:
    def test_catalog_keys_are_distinct_from_fast_and_xai(self):
        canonical = json.loads(_CANONICAL_CATALOG.read_text(encoding="utf-8"))
        bundled = json.loads(_BUNDLED_CATALOG.read_text(encoding="utf-8"))

        assert _COMPOSER_MODEL in canonical
        assert _CURSOR_GROK_MODEL in canonical
        assert canonical[_COMPOSER_MODEL] == bundled[_COMPOSER_MODEL]
        assert canonical[_CURSOR_GROK_MODEL] == bundled[_CURSOR_GROK_MODEL]
        assert "cursor_agent/composer-2.5-fast" not in canonical
        assert "composer-2.5-fast" not in canonical
        assert canonical[_COMPOSER_MODEL] != canonical["xai/grok-composer-2.5-fast"]
        assert canonical[_CURSOR_GROK_MODEL] != canonical["oa_xai/grok-4.6"]
        assert canonical[_CURSOR_GROK_MODEL] != canonical["xai/grok-4.6"]
        assert canonical[_COMPOSER_MODEL]["litellm_provider"] == "cursor_agent"
        assert canonical[_CURSOR_GROK_MODEL]["litellm_provider"] == "cursor_agent"
        assert canonical["oa_xai/grok-4.6"]["litellm_provider"] == "xai"
        assert canonical["xai/grok-4.6"]["litellm_provider"] == "xai"

    def test_catalog_rows_keep_public_list_rates_invoice_unknown(self):
        catalog = json.loads(_CANONICAL_CATALOG.read_text(encoding="utf-8"))
        composer = catalog[_COMPOSER_MODEL]["provider_specific_entry"]["cursor_agent"][
            "aawm_reference_pricing"
        ]
        cursor_grok = catalog[_CURSOR_GROK_MODEL]["provider_specific_entry"][
            "cursor_agent"
        ]["aawm_reference_pricing"]

        assert composer["actual_invoice_cost_known"] is False
        assert cursor_grok["actual_invoice_cost_known"] is False
        assert composer["rates"] == {
            "input_usd_per_million_tokens": 0.5,
            "cache_read_usd_per_million_tokens": 0.2,
            "output_usd_per_million_tokens": 2.5,
        }
        assert cursor_grok["rates"] == {
            "input_usd_per_million_tokens": 2.0,
            "cache_read_usd_per_million_tokens": 0.5,
            "output_usd_per_million_tokens": 6.0,
        }
        assert "input_cost_per_token" not in catalog[_COMPOSER_MODEL]
        assert "output_cost_per_token" not in catalog[_COMPOSER_MODEL]
        assert "input_cost_per_token" not in catalog[_CURSOR_GROK_MODEL]
        assert "output_cost_per_token" not in catalog[_CURSOR_GROK_MODEL]


class TestCursorAgentFailClosedDispatch:
    def test_alias_dispatch_does_not_route_through_cloud_agents(self):
        with pytest.raises(ProxyException) as exc_info:
            _raise_cursor_agent_alias_not_implemented(
                ingress="codex",
                candidate={
                    "provider": "cursor_agent",
                    "model": _COMPOSER_MODEL,
                    "route_family": _CODEX_CURSOR_ROUTE_FAMILY,
                },
            )
        exc = exc_info.value
        assert exc.code == "429"
        assert "not implemented for this wave" in exc.message
        assert "Cloud Agents" in exc.message
        assert exc.detail["error"]["code"] == (
            "aawm_codex_auto_agent_candidate_unavailable"
        )
