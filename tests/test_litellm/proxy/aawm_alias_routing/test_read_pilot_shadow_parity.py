"""CFG-008 test: the compiled ``read`` pilot resolves the exact common prefix
plus the mutually exclusive TUI-specific tail on both ingress projections.

``litellm/proxy/aawm_alias_config/read.yaml`` (CFG-008) no longer mirrors the
legacy ``CODEX_AAWM_LOW_CANDIDATES`` table: it carries the exact common
OpenRouter/OpenCode/Alibaba prefix and then a branch-exclusive last resort --
native Anthropic Haiku for Claude origins (``tui_attached``), or
``gpt-5.6-luna`` with authoritative ``reasoning_effort: low`` for Codex and
every non-Claude/missing/unknown origin (``tui_excluded`` keeps Luna out of
the Claude branch; ``tui_attached`` keeps Haiku out of the default branch).

Ambient state (cooldown/session-affinity dicts and the process-local active
snapshot holder) is reset before and after via the same
``clear_codex_auto_agent_alias_state``-style approach used by Wave 4's
``test_read_pilot_selection.py`` so this test cannot flap on shared,
process-global state left over from other tests.
"""

from __future__ import annotations

import datetime as dt
import os

import pytest

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    config_compiler as compiler,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import snapshot_select
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    alias_routing_state,
)

_READ_YAML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
    "litellm",
    "proxy",
    "aawm_alias_config",
    "read.yaml",
)

_REFERENCE_NOW_UTC = dt.datetime(2026, 6, 15, tzinfo=dt.timezone.utc)

# CFG-008 exact common prefix shared by every origin.
_COMMON_PREFIX = [
    ("openrouter", "openrouter/cohere/north-mini-code:free", "codex_openrouter_completion_adapter"),
    ("openrouter", "openrouter/owl-alpha", "codex_openrouter_completion_adapter"),
    ("opencode_zen", "deepseek-v4-flash", "codex_opencode_zen_adapter"),
    ("opencode_zen", "big-pickle", "codex_opencode_zen_adapter"),
    ("alibaba_token_plan", "alibaba_token_plan/qwen3.6-flash", "codex_alibaba_token_plan_chat_completions_adapter"),
]
_REMOVED_MODELS = {
    "alibaba_token_plan/qwen3.8-max-preview",
    "kimi_code/kimi-for-coding",
    "gpt-5.4-mini",
}


@pytest.fixture(autouse=True)
def _reset_alias_routing_ambient_state():
    """Neutralize shared/process-global cooldown, affinity, and snapshot state.

    Mirrors the reset approach in
    ``tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py``'s
    ``clear_codex_auto_agent_alias_state`` autouse fixture, scoped to just the
    state this shadow-parity test could otherwise flap on.
    """
    previous_snapshot = snapshot_select.get_active_routing_snapshot()
    alias_routing_state.codex.cooldown_until_monotonic_by_key.clear()
    alias_routing_state.codex.session_affinity_by_key.clear()
    yield
    alias_routing_state.codex.cooldown_until_monotonic_by_key.clear()
    alias_routing_state.codex.session_affinity_by_key.clear()
    snapshot_select.set_active_routing_snapshot(previous_snapshot)


def _compile_read_yaml():
    with open(_READ_YAML_PATH, "r", encoding="utf-8") as handle:
        raw_yaml = handle.read()
    return compiler.compile_yaml(raw_yaml)


def test_read_yaml_exists_and_compiles() -> None:
    """The read.yaml pilot config file compiles into a valid snapshot with a read alias."""
    snapshot = _compile_read_yaml()
    assert "read" in snapshot.aliases
    assert len(snapshot.aliases["read"].candidates) > 0


def test_shadow_parity_read_vs_low(monkeypatch: pytest.MonkeyPatch) -> None:
    """CFG-008 Codex/default ingress: exact common prefix + Luna low-effort tail
    for Codex and every non-Claude/missing/unknown origin."""
    snapshot = _compile_read_yaml()
    snapshot_select.set_active_routing_snapshot(snapshot)

    selected = snapshot_select._select_read_pilot_snapshot_candidates(
        client_product_label=None,
        now_utc=_REFERENCE_NOW_UTC,
    )

    selected_triples = [(c["provider"], c["model"], c["route_family"]) for c in selected]
    assert selected_triples == [
        *_COMMON_PREFIX,
        ("openai", "gpt-5.6-luna", "codex_responses"),
    ]
    assert _REMOVED_MODELS.isdisjoint({c["model"] for c in selected})
    assert selected[-1]["last_resort"] is True
    assert selected[-1]["reasoning_effort"] == "low"


def test_cfg008_claude_branch_selects_haiku_tail() -> None:
    """CFG-008 Claude origin: Anthropic ingress gets the native Haiku tail;
    the Codex ingress keeps Luna ineligible for the branch and never routes
    Haiku through Codex credentials."""
    snapshot = _compile_read_yaml()
    snapshot_select.set_active_routing_snapshot(snapshot)

    selected = snapshot_select._select_read_pilot_snapshot_candidates_anthropic(
        client_product_label="Claude/1.2",
        now_utc=_REFERENCE_NOW_UTC,
    )
    assert selected is not None
    expected_anthropic_prefix = [
        ("openrouter", "openrouter/cohere/north-mini-code:free", "anthropic_openrouter_completion_adapter"),
        ("openrouter", "openrouter/owl-alpha", "anthropic_openrouter_completion_adapter"),
        ("opencode_zen", "deepseek-v4-flash", "anthropic_opencode_zen_responses_adapter"),
        ("opencode_zen", "big-pickle", "anthropic_opencode_zen_completion_adapter"),
        ("alibaba_token_plan", "alibaba_token_plan/qwen3.6-flash", "anthropic_alibaba_token_plan_chat_completions_adapter"),
    ]
    assert [(c["provider"], c["model"], c["route_family"]) for c in selected] == [
        *expected_anthropic_prefix,
        ("anthropic", "claude-haiku-4-5-20251001", "anthropic_messages"),
    ]
    assert selected[-1]["last_resort"] is True
    assert "gpt-5.6-luna" not in {c["model"] for c in selected}

    # Codex ingress: the Anthropic-credential Haiku tail is not eligible, so
    # the Claude branch is the common prefix only (no Luna, no Haiku).
    codex_side = snapshot_select._select_read_pilot_snapshot_candidates(
        client_product_label="Claude/1.2",
        now_utc=_REFERENCE_NOW_UTC,
    )
    codex_models = [c["model"] for c in codex_side]
    assert codex_models == [triple[1] for triple in _COMMON_PREFIX]
    assert "gpt-5.6-luna" not in codex_models
    assert "claude-haiku-4-5-20251001" not in codex_models


def test_cfg008_codex_origin_selects_luna_tail() -> None:
    """CFG-008 identified Codex origin: common prefix + Luna tail; Haiku is
    ineligible on this branch."""
    snapshot = _compile_read_yaml()
    snapshot_select.set_active_routing_snapshot(snapshot)

    selected = snapshot_select._select_read_pilot_snapshot_candidates(
        client_product_label="Codex/0.31.0",
        now_utc=_REFERENCE_NOW_UTC,
    )
    models = [c["model"] for c in selected]
    assert models == [triple[1] for triple in _COMMON_PREFIX] + ["gpt-5.6-luna"]
    assert "claude-haiku-4-5-20251001" not in models
    assert selected[-1]["reasoning_effort"] == "low"


def test_cfg008_anthropic_ingress_projection_branches() -> None:
    """CFG-008 Anthropic Messages ingress projection carries the same branch
    exclusivity with the anthropic-projected route families."""
    snapshot = _compile_read_yaml()
    snapshot_select.set_active_routing_snapshot(snapshot)

    claude = snapshot_select._select_read_pilot_snapshot_candidates_anthropic(
        client_product_label="Claude/1.2",
        now_utc=_REFERENCE_NOW_UTC,
    )
    assert claude is not None
    assert [c["model"] for c in claude] == [
        triple[1] for triple in _COMMON_PREFIX
    ] + ["claude-haiku-4-5-20251001"]
    assert claude[-1]["route_family"] == "anthropic_messages"
    for candidate in claude:
        assert candidate["route_family"].startswith("anthropic_")

    default = snapshot_select._select_read_pilot_snapshot_candidates_anthropic(
        client_product_label=None,
        now_utc=_REFERENCE_NOW_UTC,
    )
    assert default is not None
    assert [c["model"] for c in default] == [
        triple[1] for triple in _COMMON_PREFIX
    ] + ["gpt-5.6-luna"]
    assert default[-1]["route_family"] == "anthropic_openai_responses_adapter"
    assert default[-1]["reasoning_effort"] == "low"
    assert "claude-haiku-4-5-20251001" not in {c["model"] for c in default}
