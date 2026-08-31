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
            "opencode_zen",
            "big-pickle",
            "codex_opencode_zen_adapter",
            50,
        ),
        ("REF", "basic-other", None, 0),
    ],
    "basic-other": [
        (
            "alibaba_token_plan",
            "alibaba_token_plan/deepseek-v4-flash-0731",
            "codex_alibaba_token_plan_chat_completions_adapter",
            100,
        ),
        (
            "zai_coding_plan",
            "zai_coding_plan/glm-5.3-flash",
            "codex_zai_coding_plan_chat_completions_adapter",
            90,
        ),
        (
            "cursor_agent",
            "cursor_agent/composer-2.5",
            "codex_cursor_agent_aiserver_adapter",
            80,
        ),
        (
            "openai",
            "gpt-5.6-luna",
            "codex_responses",
            0,
        ),
        (
            "anthropic",
            "claude-haiku-4-5-20251001",
            "anthropic_messages",
            0,
        ),
    ],
    "work": [
        ("REF", "work-other", None, 110),
        ("anthropic", "claude-sonnet-5[1m]", "anthropic_messages", 80),
        ("anthropic", "claude-sonnet-5", "anthropic_messages", 70),
        ("openai", "gpt-5.6-luna", "codex_responses", 0),
    ],
    "work-other": [
        ("REF", "sota-deepseek", None, 110),
        (
            "zai_coding_plan",
            "zai_coding_plan/glm-5.3-flash",
            "codex_zai_coding_plan_chat_completions_adapter",
            100,
        ),
        ("REF", "sota-moonshot", None, 90),
        ("REF", "sota-xai", None, 80),
    ],
    "expert": [
        ("REF", "expert-other", None, 100),
        (
            "openai",
            "gpt-5.6-terra",
            "codex_responses",
            0,
        ),
    ],
    "expert-other": [
        (
            "alibaba_token_plan",
            "alibaba_token_plan/qwen3.8-max",
            "codex_alibaba_token_plan_chat_completions_adapter",
            100,
        ),
        (
            "cursor_agent",
            "cursor_agent/cursor-grok-4.6-high",
            "codex_cursor_agent_aiserver_adapter",
            90,
        ),
        (
            "xai",
            "xai/grok-4.6",
            "codex_grok_native_responses_adapter",
            0,
        ),
    ],
    "auto-review": [
        ("REF", "auto-review-other", None, 100),
        (
            "openai",
            "gpt-5.6-luna",
            "codex_responses",
            90,
        ),
        (
            "openrouter",
            "openrouter/~deepseek/deepseek-v4-flash-latest",
            "codex_openrouter_completion_adapter",
            0,
        ),
    ],
    "auto-review-other": [
        (
            "alibaba_token_plan",
            "alibaba_token_plan/deepseek-v4-flash-0731",
            "codex_alibaba_token_plan_chat_completions_adapter",
            100,
        ),
        (
            "zai_coding_plan",
            "zai_coding_plan/glm-5.3-flash",
            "codex_zai_coding_plan_chat_completions_adapter",
            90,
        ),
        (
            "cursor_agent",
            "cursor_agent/composer-2.5",
            "codex_cursor_agent_aiserver_adapter",
            80,
        ),
    ],
    "codex-auto-review": [
        ("REF", "auto-review", None, 100),
    ],
    "sota-openai": [
        ("openai", "gpt-5.6-sol", "codex_responses", 100),
    ],
}
_WITHDRAWN_OX_ALPHA_MODELS = frozenset(
    {
        "ox-alpha-free",
        "stealth/ox-alpha",
        "openrouter/stealth/ox-alpha",
    }
)
_CONFIGURED_PROVIDER_IDS = tuple(
    sorted(REGISTERED_PROVIDERS - {"nous", "opencode_go"})
)


def _entry_identity(entry: RoutingCandidate | AliasReference) -> tuple:
    if isinstance(entry, AliasReference):
        return ("REF", entry.alias_name, None, entry.priority)
    return (entry.provider, entry.model, entry.route_family, entry.priority)


def test_compile_directory_exposes_closed_aliases_for_configured_providers() -> None:
    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    expected = tuple(
        provider_alias_name(provider_id)
        for provider_id in _CONFIGURED_PROVIDER_IDS
    )
    actual = iter_provider_alias_names(snapshot.aliases)
    assert actual == expected
    assert set(uncovered_registered_providers(snapshot.aliases)) == {
        "nous",
        "opencode_go",
    }
    assert "nvidia" in REGISTERED_PROVIDERS
    assert "provider-nvidia" in snapshot.aliases

    for provider_id in _CONFIGURED_PROVIDER_IDS:
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


_NVIDIA_CLOSED_SET = (
    (
        "nvidia",
        "nvidia/openai/gpt-oss-20b",
        "codex_nvidia_completion_adapter",
        100,
    ),
    (
        "nvidia",
        "nvidia/deepseek-ai/deepseek-v4-flash-0731",
        "codex_nvidia_completion_adapter",
        90,
    ),
    (
        "nvidia",
        "nvidia/nvidia/nemotron-3-super-120b-a12b",
        "codex_nvidia_completion_adapter",
        80,
    ),
    (
        "nvidia",
        "nvidia/minimaxai/minimax-m3",
        "codex_nvidia_completion_adapter",
        70,
    ),
    (
        "nvidia",
        "nvidia/openai/gpt-oss-120b",
        "codex_nvidia_completion_adapter",
        60,
    ),
)


def test_provider_nvidia_keeps_closed_five_model_nim_set() -> None:
    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    alias = snapshot.aliases["provider-nvidia"]
    assert alias.dispatch is None
    assert len(alias.candidates) == 5
    identities = [
        (
            entry.provider,
            entry.model,
            entry.route_family,
            entry.priority,
        )
        for entry in alias.candidates
    ]
    assert identities == list(_NVIDIA_CLOSED_SET)
    for entry in alias.candidates:
        assert isinstance(entry, RoutingCandidate)
        assert entry.provider == "nvidia"
        assert not entry.model.endswith(":free")
        assert entry.route_family == "codex_nvidia_completion_adapter"
        assert entry.anthropic_route_family is None
    assert all(not isinstance(entry, AliasReference) for entry in alias.candidates)


def test_provider_openai_keeps_cheapest_first_low_effort_order() -> None:
    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    alias = snapshot.aliases["provider-openai"]
    assert [
        (
            entry.provider,
            entry.model,
            entry.route_family,
            entry.priority,
            entry.reasoning_effort,
        )
        for entry in alias.candidates
    ] == [
        ("openai", "gpt-5.6-luna", "codex_responses", 100, "low"),
        ("openai", "gpt-5.6-terra", "codex_responses", 90, "low"),
        ("openai", "gpt-5.6-sol", "codex_responses", 0, "low"),
    ]


def test_provider_nvidia_rejects_alias_reference_and_openrouter_escape() -> None:
    escaped = """
defaults: {}
aliases:
  - name: provider-openrouter
    candidates:
      - provider: openrouter
        model: openrouter/nvidia/nemotron-super-49b:free
        route_family: codex_openrouter_completion_adapter
        priority: 100
  - name: provider-nvidia
    candidates:
      - alias_reference: provider-openrouter
        priority: 100
"""
    with pytest.raises(ConfigCompileError, match="alias_reference"):
        compiler.compile_yaml(escaped)

    crossed = """
defaults: {}
aliases:
  - name: provider-nvidia
    candidates:
      - provider: openrouter
        model: openrouter/nvidia/nemotron-super-49b:free
        route_family: codex_openrouter_completion_adapter
        priority: 100
"""
    with pytest.raises(
        (ValidationError, ConfigCompileError),
        match="provider-nvidia|expected 'nvidia'|NVIDIA|uncovered",
    ):
        compiler.compile_yaml(crossed)


def test_compiled_aliases_exclude_withdrawn_ox_alpha_routes() -> None:
    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    assert "provider-nous" not in snapshot.aliases
    assert "provider-opencode_go" not in snapshot.aliases
    for alias in snapshot.aliases.values():
        for entry in alias.candidates:
            if isinstance(entry, RoutingCandidate):
                assert entry.model not in _WITHDRAWN_OX_ALPHA_MODELS


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


def test_compile_allows_subset_of_registered_provider_aliases() -> None:
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
    snapshot = compiler.compile_yaml(raw)
    assert iter_provider_alias_names(snapshot.aliases) == ("provider-openai",)
    assert set(uncovered_registered_providers(snapshot.aliases)) == (
        set(REGISTERED_PROVIDERS) - {"openai"}
    )


def test_provider_alias_rejects_cross_provider_candidate_and_alias_reference() -> None:
    crossed = """
defaults: {}
aliases:
  - name: provider-openai
    candidates:
      - provider: openrouter
        model: openrouter/cohere/north-mini-code:free
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
    for provider_id in _CONFIGURED_PROVIDER_IDS:
        assert provider_alias_name(provider_id) in names
    assert "provider-nous" not in names
    assert "provider-opencode_go" not in names
    assert "aawm-sota-zai" not in names
