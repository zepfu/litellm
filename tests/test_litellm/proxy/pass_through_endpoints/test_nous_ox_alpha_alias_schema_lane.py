"""NOUS-003: compiled basic/work order Go, Nous, then OpenRouter ox-alpha.

Locks the numeric-gap insert of Codex-only Nous stealth/ox-alpha between
OpenCode Go and OpenRouter. Anthropic snapshots omit Go and Nous and still
include OpenRouter ox-alpha. Nous stays off the OpenRouter free-daily quota
set.

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
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
    compile_directory,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_schema import (
    CODEX_ONLY_ROUTE_FAMILIES,
)


_CODEX_GO_ROUTE_FAMILY = "codex_opencode_go_adapter"
_CODEX_NOUS_ROUTE_FAMILY = "codex_nous_chat_completions_adapter"
_GO_MODEL = "ox-alpha-free"
_NOUS_MODEL = "stealth/ox-alpha"
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


def _candidate_identity(candidate: Any) -> tuple[str, str, str]:
    return (candidate.provider, candidate.model, candidate.route_family)


def test_basic_yaml_places_go_nous_then_openrouter_ox_alpha():
    with open(_BASIC_YAML_PATH, "r", encoding="utf-8") as handle:
        snapshot = compile_yaml(handle.read())

    basic_candidates = snapshot.aliases["basic"].candidates
    first, second, third = basic_candidates[0], basic_candidates[1], basic_candidates[2]
    assert _candidate_identity(first) == (
        "opencode_go",
        _GO_MODEL,
        _CODEX_GO_ROUTE_FAMILY,
    )
    assert first.priority == 100
    assert first.anthropic_route_family is None
    assert _candidate_identity(second) == (
        "nous",
        _NOUS_MODEL,
        _CODEX_NOUS_ROUTE_FAMILY,
    )
    assert second.priority == 97
    assert second.anthropic_route_family is None
    assert _CODEX_NOUS_ROUTE_FAMILY in CODEX_ONLY_ROUTE_FAMILIES
    assert _candidate_identity(third) == (
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


def test_work_yaml_places_go_nous_then_openrouter_ox_alpha():
    snapshot = compile_directory(_ALIAS_CONFIG_DIR)
    work_candidates = snapshot.aliases["work"].candidates
    first, second, third, fourth = (
        work_candidates[0],
        work_candidates[1],
        work_candidates[2],
        work_candidates[3],
    )
    assert _candidate_identity(first) == (
        "opencode_go",
        _GO_MODEL,
        _CODEX_GO_ROUTE_FAMILY,
    )
    assert first.priority == 110
    assert first.anthropic_route_family is None
    assert _candidate_identity(second) == (
        "nous",
        _NOUS_MODEL,
        _CODEX_NOUS_ROUTE_FAMILY,
    )
    assert second.priority == 107
    assert second.anthropic_route_family is None
    assert _candidate_identity(third) == (
        "openrouter",
        _OPENROUTER_OX_ALPHA_MODEL,
        "codex_openrouter_completion_adapter",
    )
    assert third.priority == 105
    assert fourth.model == "gpt-5.3-codex-spark"


def test_anthropic_basic_skips_go_and_nous_keeps_openrouter_ox_alpha():
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
    assert _OPENROUTER_OX_ALPHA_MODEL in models
    openrouter_ox = next(
        candidate
        for candidate in selected
        if candidate["model"] == _OPENROUTER_OX_ALPHA_MODEL
    )
    assert openrouter_ox["provider"] == "openrouter"
    assert openrouter_ox["route_family"] == "anthropic_openrouter_completion_adapter"


def test_nous_lane_independent_of_openrouter_and_go():
    assert policy.CODEX_AUTO_AGENT_NOUS_LANE_KEY == "nous"
    assert policy.CODEX_AUTO_AGENT_NOUS_LANE_KEY != policy.CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY
    assert policy.CODEX_AUTO_AGENT_NOUS_LANE_KEY != policy.CODEX_AUTO_AGENT_OPENCODE_GO_LANE_KEY
    assert _NOUS_MODEL not in policy.OPENROUTER_FREE_DAILY_QUOTA_MODELS
    assert f"nous/{_NOUS_MODEL}" not in policy.OPENROUTER_FREE_DAILY_QUOTA_MODELS
    assert _OPENROUTER_OX_ALPHA_MODEL in policy.OPENROUTER_FREE_DAILY_QUOTA_MODELS


def test_direct_vs_alias_attribution():
    with open(_BASIC_YAML_PATH, "r", encoding="utf-8") as handle:
        snapshot = compile_yaml(handle.read())
    basic_candidates = snapshot.aliases["basic"].candidates
    nous = next(
        candidate
        for candidate in basic_candidates
        if getattr(candidate, "provider", None) == "nous"
        and candidate.model == _NOUS_MODEL
    )
    openrouter = next(
        candidate
        for candidate in basic_candidates
        if candidate.model == _OPENROUTER_OX_ALPHA_MODEL
    )
    assert nous.provider == "nous"
    assert nous.route_family == _CODEX_NOUS_ROUTE_FAMILY
    assert openrouter.provider == "openrouter"
    assert openrouter.route_family == "codex_openrouter_completion_adapter"
    assert nous.provider != openrouter.provider
    assert nous.model != openrouter.model
