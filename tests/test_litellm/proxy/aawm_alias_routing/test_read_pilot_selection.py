"""RED-phase tests for Wave 4: selector integration for the ``read`` pilot only.

These tests target the new selection surface Wave 4 must add to
``llm_passthrough_endpoints.py`` (candidate resolution for ``read`` from the
compiled snapshot; priority/tie/TUI/schedule selection semantics) while
asserting every other alias keeps resolving from the untouched hard-coded
``policy.py`` tables, and that NO new routing-decision recording path is
added (per operator: routing decisions are not persisted beyond existing
``session_history`` as-routed fields).

Assumed new surface (engineer's contract to satisfy, per Wave 4 Source Spec):
  - ``lpe.get_active_routing_snapshot()`` / ``lpe.set_active_routing_snapshot(snapshot)``
    — process-local snapshot holder accessor/setter (wired to config_snapshot.py).
  - ``lpe._get_codex_auto_agent_candidates_for_alias("read", ...)`` resolves from
    the active snapshot instead of ``CODEX_AAWM_LOW_CANDIDATES``.
  - ``lpe._order_snapshot_candidates_by_priority(candidates)`` — pure ordering.
  - ``lpe._select_proportional_snapshot_candidate(candidates, weights, rng)`` — pure tie-break.
  - ``lpe._is_tui_attached_candidate_eligible(candidate, client_product_label)`` — pure TUI gate.
  - ``lpe._is_snapshot_candidate_in_schedule_window(candidate, now_utc)`` — pure schedule gate.

Modules under test do not exist yet (``config_snapshot``, ``config_compiler``),
so import failure at collection time is the correct red-phase signal.
"""

from __future__ import annotations

import inspect

import pytest

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (  # type: ignore[import-not-found]
    config_compiler as compiler,
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
    previous = lpe.get_active_routing_snapshot()
    lpe.set_active_routing_snapshot(snapshot)
    yield snapshot
    lpe.set_active_routing_snapshot(previous)


def test_read_alias_uses_snapshot(snapshot_fixture) -> None:
    """``read`` resolves candidates from the compiled snapshot, not CODEX_AAWM_LOW_CANDIDATES."""
    candidates = lpe._get_codex_auto_agent_candidates_for_alias("read")
    models = [c["model"] for c in candidates]
    assert "openrouter/snapshot-only-model" in models
    low_models = [c["model"] for c in CODEX_AAWM_LOW_CANDIDATES]
    assert models != low_models


def test_other_aliases_unchanged(snapshot_fixture) -> None:
    """aawm-read, aawm-low, aawm-sota still resolve from the hard-coded tables."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import policy

    assert lpe._get_codex_auto_agent_candidates_for_alias(CODEX_AAWM_READ_ALIAS) == policy.CODEX_AUTO_AGENT_CANDIDATES
    assert lpe._get_codex_auto_agent_candidates_for_alias(CODEX_AAWM_LOW_ALIAS) == CODEX_AAWM_LOW_CANDIDATES
    assert lpe._get_codex_auto_agent_candidates_for_alias(CODEX_AAWM_SOTA_ALIAS) == policy.CODEX_AAWM_SOTA_CANDIDATES


def test_priority_descending_selection(snapshot_fixture) -> None:
    """Higher priority first; priority:0 only when all others are cooled/ineligible."""
    read_alias = snapshot_fixture.aliases["read"]
    ordered = lpe._order_snapshot_candidates_by_priority(read_alias.candidates)
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
        selected = lpe._select_proportional_snapshot_candidate(candidates, weights, rng)
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
    eligible = [c for c in candidates if lpe._is_tui_attached_candidate_eligible(c, client_product_label=None)]
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
    assert lpe._is_tui_attached_candidate_eligible(candidate, client_product_label="Claude/1.2")
    assert not lpe._is_tui_attached_candidate_eligible(candidate, client_product_label="Codex/1.0")


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

    assert lpe._is_snapshot_candidate_in_schedule_window(promo_candidate, now_utc=now_within_window)
    assert not lpe._is_snapshot_candidate_in_schedule_window(promo_candidate, now_utc=now_after_window)

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
