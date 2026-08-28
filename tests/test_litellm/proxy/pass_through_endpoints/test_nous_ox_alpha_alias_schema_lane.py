"""CFG-035/038: operational alias replacement and retained Nous lane policy.

Operational ``basic`` and ``work`` aliases no longer contain the withdrawn
OX-alpha block. The ``work`` alias declares the shared CFG-035 graph and
inherits the canonical Grok order through ``work-other`` and ``sota-xai``.

No provider egress, no live Hermes reads.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    policy,
    snapshot_select,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_compiler import (
    compile_yaml,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_snapshot import (
    AliasReference,
    RoutingCandidate,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
    compile_directory,
)


_CODEX_ZAI_ROUTE_FAMILY = "codex_zai_coding_plan_chat_completions_adapter"
_ZAI_MODEL = "zai_coding_plan/glm-5.3-flash"
_NOUS_MODEL = "stealth/ox-alpha"
_OPENROUTER_OX_ALPHA_MODEL = "openrouter/stealth/ox-alpha"
_LEGACY_OX_MODELS = frozenset(
    {
        "ox-alpha-free",
        _NOUS_MODEL,
        _OPENROUTER_OX_ALPHA_MODEL,
    }
)
_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
_BASIC_YAML_PATH = os.path.join(
    _REPO_ROOT, "litellm", "proxy", "aawm_alias_config", "basic.yaml"
)
_ALIAS_CONFIG_DIR = Path(_REPO_ROOT) / "litellm" / "proxy" / "aawm_alias_config"


def _candidate_identity(candidate: Any) -> tuple[str, str, str]:
    return (candidate.provider, candidate.model, candidate.route_family)


def test_basic_yaml_replaces_legacy_ox_alpha_block_with_zai_flash():
    with open(_BASIC_YAML_PATH, "r", encoding="utf-8") as handle:
        snapshot = compile_yaml(handle.read())

    basic_candidates = snapshot.aliases["basic"].candidates
    first = basic_candidates[0]
    assert _candidate_identity(first) == (
        "zai_coding_plan",
        _ZAI_MODEL,
        _CODEX_ZAI_ROUTE_FAMILY,
    )
    assert first.priority == 100
    assert first.reasoning_effort == "low"
    assert not any(
        candidate.model in _LEGACY_OX_MODELS for candidate in basic_candidates
    )

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
    assert cohere_index == 1
    assert basic_candidates[cohere_index - 1].model == _ZAI_MODEL


def test_work_yaml_compiles_shared_cfg035_cfg038_graph():
    snapshot = compile_directory(_ALIAS_CONFIG_DIR)
    entries = snapshot.aliases["work"].candidates
    assert len(entries) == 6
    assert [
        (
            ("REF", entry.alias_name, None, entry.priority)
            if isinstance(entry, AliasReference)
            else (entry.provider, entry.model, entry.route_family, entry.priority)
        )
        for entry in entries
    ] == [
        (
            "zai_coding_plan",
            _ZAI_MODEL,
            _CODEX_ZAI_ROUTE_FAMILY,
            110,
        ),
        ("openai", "gpt-5.3-codex-spark", "codex_responses", 100),
        ("REF", "work-other", None, 90),
        ("anthropic", "claude-sonnet-5[1m]", "anthropic_messages", 80),
        ("anthropic", "claude-sonnet-5", "anthropic_messages", 70),
        ("openai", "gpt-5.6-luna", "codex_responses", 0),
    ]
    for candidate in entries[3:5]:
        assert isinstance(candidate, RoutingCandidate)
        assert candidate.anthropic_route_family == "anthropic_messages"
        assert candidate.reasoning_effort == "max"
        assert candidate.tui_attached == "Claude"
    luna = entries[-1]
    assert isinstance(luna, RoutingCandidate)
    assert luna.reasoning_effort == "max"
    assert not any(
        getattr(candidate, "model", None) in _LEGACY_OX_MODELS
        for candidate in entries
    )


def test_anthropic_basic_excludes_legacy_ox_alpha_candidates():
    with open(_BASIC_YAML_PATH, "r", encoding="utf-8") as handle:
        snapshot = compile_yaml(handle.read())
    previous = snapshot_select.get_active_routing_snapshot()
    snapshot_select.set_active_routing_snapshot(snapshot)
    try:
        selected = snapshot_select._select_snapshot_candidates(
            "basic",
            ingress="anthropic",
        )
    finally:
        snapshot_select.set_active_routing_snapshot(previous)

    providers = [candidate["provider"] for candidate in selected]
    models = [candidate["model"] for candidate in selected]
    assert "opencode_go" not in providers
    assert "nous" not in providers
    assert "openrouter" in providers
    assert "zai_coding_plan" not in providers
    assert not any(model in _LEGACY_OX_MODELS for model in models)


def test_nous_lane_independent_of_openrouter_and_go():
    assert policy.CODEX_AUTO_AGENT_NOUS_LANE_KEY == "nous"
    assert policy.CODEX_AUTO_AGENT_NOUS_LANE_KEY != policy.CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY
    assert policy.CODEX_AUTO_AGENT_NOUS_LANE_KEY != policy.CODEX_AUTO_AGENT_OPENCODE_GO_LANE_KEY
    assert _NOUS_MODEL not in policy.OPENROUTER_FREE_DAILY_QUOTA_MODELS
    assert f"nous/{_NOUS_MODEL}" not in policy.OPENROUTER_FREE_DAILY_QUOTA_MODELS
    assert _OPENROUTER_OX_ALPHA_MODEL in policy.OPENROUTER_FREE_DAILY_QUOTA_MODELS


def test_operational_aliases_exclude_legacy_ox_alpha_models():
    snapshot = compile_directory(_ALIAS_CONFIG_DIR)
    for alias_name in ("basic", "work"):
        alias = snapshot.aliases[alias_name]
        assert not any(
            getattr(candidate, "model", None) in _LEGACY_OX_MODELS
            for candidate in alias.candidates
        )
