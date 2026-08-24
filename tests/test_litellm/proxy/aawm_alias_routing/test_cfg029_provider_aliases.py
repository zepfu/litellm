"""CFG-029: provider-pinned aliases compile as closed same-provider sets."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    config_compiler as compiler,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_compiler import (
    ConfigCompileError,
    iter_provider_alias_names,
    provider_alias_name,
    uncovered_registered_providers,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_schema import (
    REGISTERED_PROVIDERS,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_snapshot import (
    AliasReference,
    RoutingCandidate,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
    DEFAULT_CONFIG_DIR,
    compile_directory,
)


_OPERATIONAL_ALIAS_ORDER = {
    "basic": [
        ("opencode_go", "ox-alpha-free", "codex_opencode_go_adapter", 100),
        ("nous", "stealth/ox-alpha", "codex_nous_chat_completions_adapter", 97),
        (
            "openrouter",
            "openrouter/stealth/ox-alpha",
            "codex_openrouter_completion_adapter",
            95,
        ),
        (
            "cohere",
            "cohere/north-mini-code-1-0",
            "codex_cohere_chat_completions_adapter",
            90,
        ),
        (
            "openrouter",
            "openrouter/cohere/north-mini-code:free",
            "codex_openrouter_completion_adapter",
            80,
        ),
        (
            "openrouter",
            "openrouter/owl-alpha",
            "codex_openrouter_completion_adapter",
            70,
        ),
        (
            "opencode_zen",
            "deepseek-v4-flash-free",
            "codex_opencode_zen_adapter",
            60,
        ),
        ("opencode_zen", "big-pickle", "codex_opencode_zen_adapter", 50),
        (
            "alibaba_token_plan",
            "alibaba_token_plan/deepseek-v4-flash-0731",
            "codex_alibaba_token_plan_chat_completions_adapter",
            45,
        ),
        (
            "cursor_agent",
            "cursor_agent/composer-2.5",
            "codex_cursor_agent_aiserver_adapter",
            42,
        ),
        (
            "alibaba_token_plan",
            "alibaba_token_plan/qwen3.6-flash",
            "codex_alibaba_token_plan_chat_completions_adapter",
            40,
        ),
        ("openai", "gpt-5.6-luna", "codex_responses", 0),
        ("anthropic", "claude-haiku-4-5-20251001", "anthropic_messages", 0),
    ],
    "work": [
        ("opencode_go", "ox-alpha-free", "codex_opencode_go_adapter", 110),
        ("nous", "stealth/ox-alpha", "codex_nous_chat_completions_adapter", 107),
        (
            "openrouter",
            "openrouter/stealth/ox-alpha",
            "codex_openrouter_completion_adapter",
            105,
        ),
        ("openai", "gpt-5.3-codex-spark", "codex_responses", 100),
        ("REF", "work-other", None, 90),
        ("anthropic", "claude-sonnet-5[1m]", "anthropic_messages", 80),
        ("anthropic", "claude-sonnet-5", "anthropic_messages", 70),
        ("openai", "gpt-5.6-luna", "codex_responses", 0),
    ],
    "work-other": [
        ("REF", "sota-deepseek", None, 110),
        ("REF", "sota-moonshot", None, 100),
        ("REF", "sota-xai", None, 90),
    ],
    "expert": [
        (
            "alibaba_token_plan",
            "alibaba_token_plan/qwen3.8-max",
            "codex_alibaba_token_plan_chat_completions_adapter",
            110,
        ),
        ("anthropic", "claude-opus-5", "anthropic_messages", 100),
        ("openai", "gpt-5.6-terra", "codex_responses", 0),
    ],
    "sota-openai": [
        ("openai", "gpt-5.6-sol", "codex_responses", 100),
    ],
    "sota-xai": [
        ("xai", "oa_xai/grok-4.6", "codex_xai_oauth_responses_adapter", 100),
        (
            "cursor_agent",
            "cursor_agent/cursor-grok-4.6-high",
            "codex_cursor_agent_aiserver_adapter",
            90,
        ),
    ],
    "sota-alibaba": [
        (
            "alibaba_token_plan",
            "alibaba_token_plan/qwen3.8-max",
            "codex_alibaba_token_plan_chat_completions_adapter",
            100,
        ),
        (
            "alibaba_token_plan",
            "alibaba_token_plan/qwen3.7-max",
            "codex_alibaba_token_plan_chat_completions_adapter",
            0,
        ),
    ],
    "sota-moonshot": [
        (
            "kimi_code",
            "kimi_code/k3",
            "codex_kimi_chat_completions_adapter",
            100,
        ),
    ],
    "sota-deepseek": [
        (
            "alibaba_token_plan",
            "alibaba_token_plan/deepseek-v4-pro",
            "codex_alibaba_token_plan_chat_completions_adapter",
            100,
        ),
    ],
    "sota-zai": [
        (
            "zai_coding_plan",
            "zai_coding_plan/glm-5.3",
            "codex_zai_coding_plan_chat_completions_adapter",
            110,
        ),
        (
            "alibaba_token_plan",
            "alibaba_token_plan/glm-5.2",
            "codex_alibaba_token_plan_chat_completions_adapter",
            100,
        ),
    ],
}


def _entry_identity(entry: RoutingCandidate | AliasReference) -> tuple:
    if isinstance(entry, AliasReference):
        return ("REF", entry.alias_name, None, entry.priority)
    return (entry.provider, entry.model, entry.route_family, entry.priority)


def test_compile_directory_exposes_one_closed_provider_alias_per_registered_provider() -> None:
    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    expected = tuple(
        provider_alias_name(provider_id)
        for provider_id in sorted(REGISTERED_PROVIDERS)
    )
    actual = iter_provider_alias_names(snapshot.aliases)
    assert actual == expected
    assert uncovered_registered_providers(snapshot.aliases) == ()
    assert "nvidia" not in REGISTERED_PROVIDERS
    assert "provider-nvidia" not in snapshot.aliases

    for provider_id in REGISTERED_PROVIDERS:
        name = provider_alias_name(provider_id)
        alias = snapshot.aliases[name]
        assert alias.dispatch is None
        assert alias.candidates, f"{name} has no candidates"
        for entry in alias.candidates:
            assert isinstance(entry, RoutingCandidate), (
                f"{name} must not alias_reference {entry!r}"
            )
            assert entry.provider == provider_id
            assert entry.model
            assert entry.route_family


def test_provider_opencode_zen_keeps_both_adapter_forms_distinct() -> None:
    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    alias = snapshot.aliases["provider-opencode_zen"]
    pairs = [
        (entry.model, entry.route_family, entry.anthropic_route_family)
        for entry in alias.candidates
    ]
    assert (
        "deepseek-v4-flash-free",
        "codex_opencode_zen_adapter",
        "anthropic_opencode_zen_responses_adapter",
    ) in pairs
    assert (
        "big-pickle",
        "codex_opencode_zen_adapter",
        "anthropic_opencode_zen_completion_adapter",
    ) in pairs
    assert pairs[0] != pairs[1]


def test_provider_xai_keeps_managed_and_native_lanes_distinct() -> None:
    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    alias = snapshot.aliases["provider-xai"]
    pairs = [
        (entry.model, entry.route_family, entry.anthropic_route_family)
        for entry in alias.candidates
    ]
    assert (
        "oa_xai/grok-4.6",
        "codex_xai_oauth_responses_adapter",
        "anthropic_xai_oauth_responses_adapter",
    ) in pairs
    assert (
        "xai/grok-4.6",
        "codex_grok_native_responses_adapter",
        "anthropic_grok_native_responses_adapter",
    ) in pairs
    assert all(entry.provider == "xai" for entry in alias.candidates)


def test_provider_anthropic_stays_anthropic_native_without_tui_gate() -> None:
    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    alias = snapshot.aliases["provider-anthropic"]
    assert alias.candidates
    for entry in alias.candidates:
        assert entry.provider == "anthropic"
        assert entry.route_family == "anthropic_messages"
        assert entry.anthropic_route_family == "anthropic_messages"
        assert entry.tui_attached is None
        assert entry.tui_excluded is None


def test_operational_alias_candidate_order_is_unchanged_by_provider_aliases() -> None:
    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    for name, expected in _OPERATIONAL_ALIAS_ORDER.items():
        actual = [_entry_identity(entry) for entry in snapshot.aliases[name].candidates]
        assert actual == expected, name
    sota = snapshot.aliases["sota"]
    assert sota.dispatch is not None
    assert sota.candidates == ()
    assert "provider-openai" not in {
        rule.target_alias for rule in sota.dispatch.by_tui
    }
    assert sota.dispatch.default != "provider-openai"


def test_compile_reports_uncovered_registered_provider_when_any_provider_alias_exists() -> None:
    raw = """
defaults: {}
aliases:
  - name: provider-openai
    candidates:
      - provider: openai
        model: gpt-5.6-sol
        route_family: codex_responses
        priority: 100
"""
    with pytest.raises(ConfigCompileError, match="uncovered registered providers"):
        compiler.compile_yaml(raw)


def test_provider_alias_rejects_cross_provider_candidate_and_alias_reference() -> None:
    crossed = """
defaults: {}
aliases:
  - name: provider-openai
    candidates:
      - provider: openrouter
        model: openrouter/owl-alpha
        route_family: codex_openrouter_completion_adapter
        priority: 100
"""
    with pytest.raises(
        (ValidationError, ConfigCompileError),
        match="provider-openai|uncovered|expected 'openai'",
    ):
        compiler.compile_yaml(crossed)

    escaped = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openai
        model: gpt-5.6-sol
        route_family: codex_responses
        priority: 100
  - name: provider-openai
    candidates:
      - alias_reference: basic
        priority: 100
"""
    with pytest.raises(ConfigCompileError, match="alias_reference"):
        compiler.compile_yaml(escaped)


def test_iter_compiled_alias_names_includes_live_provider_aliases() -> None:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.catalog import (
        iter_compiled_alias_names,
    )

    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    names = iter_compiled_alias_names(snapshot)
    for provider_id in REGISTERED_PROVIDERS:
        assert provider_alias_name(provider_id) in names
    assert "aawm-sota-zai" not in names
