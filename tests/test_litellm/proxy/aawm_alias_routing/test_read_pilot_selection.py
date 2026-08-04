"""RED-phase tests for Wave 4: selector integration for the ``read`` pilot only.

These tests target the selection surface Wave 4 added for candidate resolution
from the compiled snapshot; priority/tie/TUI/schedule selection semantics;
unchanged hard-coded ``policy.py`` tables for every other alias; and no new
routing-decision recording path beyond existing ``session_history`` as-routed
fields.

The extracted selection surface is owned by
``aawm_alias_routing.snapshot_select``.
"""

from __future__ import annotations

import inspect

import pytest

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    config_compiler as compiler,
    snapshot_select,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.policy import (
    CODEX_AAWM_LOW_CANDIDATES,
    CODEX_AAWM_READ_ALIAS,
    CODEX_AAWM_SOTA_ALIAS,
    CODEX_AAWM_LOW_ALIAS,
)


_SNAPSHOT_YAML = """
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openrouter
        model: openrouter/snapshot-only-model
        route_family: codex_openrouter_completion_adapter
        priority: 900
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 0
"""


@pytest.fixture()
def snapshot_fixture():
    snapshot = compiler.compile_yaml(_SNAPSHOT_YAML)
    previous = snapshot_select.get_active_routing_snapshot()
    snapshot_select.set_active_routing_snapshot(snapshot)
    yield snapshot
    snapshot_select.set_active_routing_snapshot(previous)


def test_read_alias_uses_snapshot(snapshot_fixture) -> None:
    """``read`` resolves candidates from the compiled snapshot, not CODEX_AAWM_LOW_CANDIDATES."""
    candidates = snapshot_select._get_codex_auto_agent_candidates_for_alias("read")
    models = [c["model"] for c in candidates]
    assert "openrouter/snapshot-only-model" in models
    low_models = [c["model"] for c in CODEX_AAWM_LOW_CANDIDATES]
    assert models != low_models


def test_other_aliases_unchanged(snapshot_fixture) -> None:
    """aawm-read, aawm-low, aawm-sota still resolve from the hard-coded tables."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import policy

    assert snapshot_select._get_codex_auto_agent_candidates_for_alias(CODEX_AAWM_READ_ALIAS) == policy.CODEX_AUTO_AGENT_CANDIDATES
    assert snapshot_select._get_codex_auto_agent_candidates_for_alias(CODEX_AAWM_LOW_ALIAS) == CODEX_AAWM_LOW_CANDIDATES
    assert snapshot_select._get_codex_auto_agent_candidates_for_alias(CODEX_AAWM_SOTA_ALIAS) == policy.CODEX_AAWM_SOTA_CANDIDATES


def test_priority_descending_selection(snapshot_fixture) -> None:
    """Higher priority first; priority:0 only when all others are cooled/ineligible."""
    read_alias = snapshot_fixture.aliases["read"]
    ordered = snapshot_select._order_snapshot_candidates_by_priority(read_alias.candidates)
    assert ordered[0].model == "openrouter/snapshot-only-model"
    assert ordered[-1].priority == 0
    assert ordered[-1].model == "gpt-5.4-mini"


def test_proportional_tie_distribution() -> None:
    """Equal-priority candidates split by weight over many selections within tolerance."""
    raw = """
defaults: {}
aliases:
  - name: read
    distribution_strategy: proportional
    candidates:
      - provider: openrouter
        model: a
        route_family: codex_openrouter_completion_adapter
        priority: 50
        weight: 1
      - provider: openrouter
        model: b
        route_family: codex_openrouter_completion_adapter
        priority: 50
        weight: 3
"""
    snapshot = compiler.compile_yaml(raw)
    candidates = snapshot.aliases["read"].candidates
    weights = {c.model: c.weight for c in candidates}

    import random

    rng = random.Random(1234)
    counts: dict[str, int] = {"a": 0, "b": 0}
    n_trials = 4000
    for _ in range(n_trials):
        selected = snapshot_select._select_proportional_snapshot_candidate(candidates, weights, rng)
        counts[selected.model] += 1

    ratio_a = counts["a"] / n_trials
    ratio_b = counts["b"] / n_trials
    assert abs(ratio_a - 0.25) < 0.05
    assert abs(ratio_b - 0.75) < 0.05


def test_tui_attached_excluded_on_unknown_tui() -> None:
    """With no client-product label, a tui_attached candidate is skipped; alias still resolves."""
    raw = """
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openai
        model: claude-only-model
        route_family: codex_responses
        priority: 100
        tui_attached: Claude
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 0
"""
    snapshot = compiler.compile_yaml(raw)
    candidates = snapshot.aliases["read"].candidates
    eligible = [c for c in candidates if snapshot_select._is_tui_attached_candidate_eligible(c, client_product_label=None)]
    eligible_models = [c.model for c in eligible]
    assert "claude-only-model" not in eligible_models
    assert "gpt-5.4-mini" in eligible_models


def test_tui_attached_selected_when_identified() -> None:
    """With Claude/x.y present, the tui_attached: Claude candidate is eligible."""
    raw = """
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openai
        model: claude-only-model
        route_family: codex_responses
        priority: 100
        tui_attached: Claude
"""
    snapshot = compiler.compile_yaml(raw)
    candidate = snapshot.aliases["read"].candidates[0]
    assert snapshot_select._is_tui_attached_candidate_eligible(candidate, client_product_label="Claude/1.2")
    assert not snapshot_select._is_tui_attached_candidate_eligible(candidate, client_product_label="Codex/1.0")


def test_tui_excluded_gate_version_insensitive() -> None:
    """CFG-008: tui_excluded gates only the matching identified product name."""
    raw = """
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openai
        model: gpt-5.6-luna
        route_family: codex_responses
        priority: 0
        tui_excluded: Claude
"""
    snapshot = compiler.compile_yaml(raw)
    candidate = snapshot.aliases["read"].candidates[0]
    gate = snapshot_select._is_tui_excluded_candidate_eligible
    # Claude origin (any version) is excluded.
    assert not gate(candidate, client_product_label="Claude/1.2")
    assert not gate(candidate, client_product_label="Claude/9.9")
    # Other identified origins and missing/unknown origins remain eligible.
    assert gate(candidate, client_product_label="Codex/1.0")
    assert gate(candidate, client_product_label="Grok/0.1")
    assert gate(candidate, client_product_label=None)


def test_mutually_exclusive_tails_branch_selection() -> None:
    """CFG-008: tui_attached + tui_excluded yield exactly one eligible tail."""
    raw = """
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openrouter
        model: openrouter/owl-alpha
        route_family: codex_openrouter_completion_adapter
        priority: 80
      - provider: openai
        model: gpt-5.6-luna
        route_family: codex_responses
        priority: 0
        reasoning_effort: low
        tui_excluded: Claude
      - provider: anthropic
        model: claude-haiku-4-5-20251001
        route_family: anthropic_messages
        anthropic_route_family: anthropic_messages
        priority: 0
        tui_attached: Claude
"""
    snapshot = compiler.compile_yaml(raw)
    previous = snapshot_select.get_active_routing_snapshot()
    snapshot_select.set_active_routing_snapshot(snapshot)
    try:
        # Codex ingress, Claude origin: Luna is excluded for the Claude branch
        # and the Anthropic-credential Haiku tail is Anthropic-ingress-only,
        # so only the common prefix remains (fail closed past it).
        claude_codex = snapshot_select._select_read_pilot_snapshot_candidates(
            client_product_label="Claude/1.2",
        )
        assert [c["model"] for c in claude_codex] == ["openrouter/owl-alpha"]

        # Anthropic ingress, Claude origin: native Haiku tail, Luna ineligible.
        claude_anthropic = snapshot_select._select_read_pilot_snapshot_candidates_anthropic(
            client_product_label="Claude/1.2",
        )
        assert claude_anthropic is not None
        assert [c["model"] for c in claude_anthropic] == [
            "openrouter/owl-alpha",
            "claude-haiku-4-5-20251001",
        ]
        assert claude_anthropic[-1]["last_resort"] is True
        assert claude_anthropic[-1]["route_family"] == "anthropic_messages"

        for label in (None, "Codex/0.31", "SomeUnknownTUI/2.0"):
            default = snapshot_select._select_read_pilot_snapshot_candidates(
                client_product_label=label,
            )
            default_models = [c["model"] for c in default]
            assert default_models == ["openrouter/owl-alpha", "gpt-5.6-luna"], label
            assert default[-1]["last_resort"] is True
            assert default[-1]["reasoning_effort"] == "low"
    finally:
        snapshot_select.set_active_routing_snapshot(previous)


def test_schedule_window_close_stops_new_affinity() -> None:
    """After a window closes, no NEW affinity to the out-of-window model; existing continues."""
    import datetime as dt

    raw = """
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: alibaba_token_plan
        model: alibaba_token_plan/qwen3.8-max-preview
        route_family: codex_alibaba_token_plan_chat_completions_adapter
        priority: 900
        schedule:
          start: "2026-07-01T00:00:00Z"
          end: "2026-07-15T00:00:00Z"
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 0
"""
    snapshot = compiler.compile_yaml(raw)
    promo_candidate = snapshot.aliases["read"].candidates[0]

    now_within_window = dt.datetime(2026, 7, 5, tzinfo=dt.timezone.utc)
    now_after_window = dt.datetime(2026, 7, 20, tzinfo=dt.timezone.utc)

    assert snapshot_select._is_snapshot_candidate_in_schedule_window(promo_candidate, now_utc=now_within_window)
    assert not snapshot_select._is_snapshot_candidate_in_schedule_window(promo_candidate, now_utc=now_after_window)

    # An existing affinity-pinned session must continue on the out-of-window model —
    # the schedule gate only prevents NEW affinity, it does not evict existing state.
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
        AliasFamilyState,
    )

    family_state = AliasFamilyState()
    family_state.set_affinity_memory(
        "session-key-1",
        {
            "provider": promo_candidate.provider,
            "model": promo_candidate.model,
            "route_family": promo_candidate.route_family,
            "last_resort": False,
        },
        ttl_seconds=3600.0,
    )
    existing_affinity = family_state.get_affinity_memory("session-key-1")
    assert existing_affinity is not None
    assert existing_affinity["model"] == promo_candidate.model


def test_no_new_routing_decision_recording(snapshot_fixture) -> None:
    """Selecting a read candidate adds no new session_history write path / hash persistence."""
    # This contract intentionally inspects the facade module itself.
    source = inspect.getsource(lpe)
    # No config-hash/version persistence path introduced by the pilot.
    assert "config_hash" not in _extract_session_history_write_regions(source)
    assert "config_version" not in _extract_session_history_write_regions(source)
    # No new session_history-recording function specific to the read pilot.
    assert not hasattr(lpe, "_record_read_pilot_routing_decision")
    assert not hasattr(lpe, "_persist_routing_snapshot_selection")


def _extract_session_history_write_regions(source: str) -> str:
    """Best-effort extraction of session_history write call regions from source text."""
    regions: list[str] = []
    marker = "inbound_model_alias"
    idx = 0
    while True:
        pos = source.find(marker, idx)
        if pos == -1:
            break
        regions.append(source[max(0, pos - 400) : pos + 400])
        idx = pos + len(marker)
    return "\n".join(regions)
