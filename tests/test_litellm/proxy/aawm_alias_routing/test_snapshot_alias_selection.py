"""Positive tests for generic snapshot-backed alias selection."""

from __future__ import annotations

import datetime as dt
import random

import pytest

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    config_compiler as compiler,
    snapshot_select,
)


_SNAPSHOT_YAML = """
defaults: {}
aliases:
  - name: snapshot-alias
    candidates:
      - provider: openrouter
        model: openrouter/snapshot-only-model
        route_family: codex_openrouter_completion_adapter
        anthropic_route_family: anthropic_openrouter_completion_adapter
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


def test_alias_lookup_and_both_ingress_projections_use_snapshot(
    snapshot_fixture,
) -> None:
    canonical_alias = snapshot_select._lookup_active_snapshot_canonical_alias(
        "  SNAPSHOT-ALIAS  "
    )
    assert canonical_alias == "snapshot-alias"

    codex = snapshot_select._select_snapshot_candidates(
        canonical_alias,
        ingress="codex",
    )
    anthropic = snapshot_select._select_snapshot_candidates(
        canonical_alias,
        ingress="anthropic",
    )

    assert [candidate["model"] for candidate in codex] == [
        "openrouter/snapshot-only-model",
        "gpt-5.4-mini",
    ]
    assert [candidate["model"] for candidate in anthropic] == [
        "openrouter/snapshot-only-model",
        "gpt-5.4-mini",
    ]
    assert codex[0]["route_family"] == "codex_openrouter_completion_adapter"
    assert (
        anthropic[0]["route_family"]
        == "anthropic_openrouter_completion_adapter"
    )


def test_priority_descending_selection(snapshot_fixture) -> None:
    """Higher priority first; priority:0 only when all others are cooled/ineligible."""
    alias = snapshot_fixture.aliases["snapshot-alias"]
    ordered = snapshot_select._order_snapshot_candidates_by_priority(
        alias.candidates
    )
    assert ordered[0].model == "openrouter/snapshot-only-model"
    assert ordered[-1].priority == 0
    assert ordered[-1].model == "gpt-5.4-mini"


def test_proportional_tie_distribution() -> None:
    """Equal-priority candidates split by weight over many selections within tolerance."""
    raw = """
defaults: {}
aliases:
  - name: weighted-alias
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
    candidates = snapshot.aliases["weighted-alias"].candidates
    weights = {c.model: c.weight for c in candidates}

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
  - name: tui-alias
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
    candidates = snapshot.aliases["tui-alias"].candidates
    eligible = [c for c in candidates if snapshot_select._is_tui_attached_candidate_eligible(c, client_product_label=None)]
    eligible_models = [c.model for c in eligible]
    assert "claude-only-model" not in eligible_models
    assert "gpt-5.4-mini" in eligible_models


def test_tui_attached_selected_when_identified() -> None:
    """With Claude/x.y present, the tui_attached: Claude candidate is eligible."""
    raw = """
defaults: {}
aliases:
  - name: tui-alias
    candidates:
      - provider: openai
        model: claude-only-model
        route_family: codex_responses
        priority: 100
        tui_attached: Claude
"""
    snapshot = compiler.compile_yaml(raw)
    candidate = snapshot.aliases["tui-alias"].candidates[0]
    assert snapshot_select._is_tui_attached_candidate_eligible(candidate, client_product_label="Claude/1.2")
    assert not snapshot_select._is_tui_attached_candidate_eligible(candidate, client_product_label="Codex/1.0")


def test_tui_excluded_gate_version_insensitive() -> None:
    """CFG-008: tui_excluded gates only the matching identified product name."""
    raw = """
defaults: {}
aliases:
  - name: tui-alias
    candidates:
      - provider: openai
        model: gpt-5.6-luna
        route_family: codex_responses
        priority: 0
        tui_excluded: Claude
"""
    snapshot = compiler.compile_yaml(raw)
    candidate = snapshot.aliases["tui-alias"].candidates[0]
    gate = snapshot_select._is_tui_excluded_candidate_eligible
    # Claude origin (any version) is excluded.
    assert not gate(candidate, client_product_label="Claude/1.2")
    assert not gate(candidate, client_product_label="Claude/9.9")
    # Other identified origins and missing/unknown origins remain eligible.
    assert gate(candidate, client_product_label="Codex/1.0")
    assert gate(candidate, client_product_label="Grok/0.1")
    assert gate(candidate, client_product_label=None)


def test_mutually_exclusive_tui_branches() -> None:
    """TUI attachment and exclusion produce one eligible branch."""
    raw = """
defaults: {}
aliases:
  - name: branch-alias
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
        claude_codex = snapshot_select._select_snapshot_candidates(
            "branch-alias",
            ingress="codex",
            client_product_label="Claude/1.2",
        )
        assert [c["model"] for c in claude_codex] == ["openrouter/owl-alpha"]

        # Anthropic ingress, Claude origin: native Haiku tail, Luna ineligible.
        claude_anthropic = snapshot_select._select_snapshot_candidates(
            "branch-alias",
            ingress="anthropic",
            client_product_label="Claude/1.2",
        )
        assert [c["model"] for c in claude_anthropic] == [
            "openrouter/owl-alpha",
            "claude-haiku-4-5-20251001",
        ]
        assert claude_anthropic[-1]["last_resort"] is True
        assert claude_anthropic[-1]["route_family"] == "anthropic_messages"

        for label in (None, "Codex/0.31", "SomeUnknownTUI/2.0"):
            default = snapshot_select._select_snapshot_candidates(
                "branch-alias",
                ingress="codex",
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
    raw = """
defaults: {}
aliases:
  - name: scheduled-alias
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
    promo_candidate = snapshot.aliases["scheduled-alias"].candidates[0]

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


def test_ordinary_alias_is_selectable_directly_and_through_reference() -> None:
    snapshot = compiler.compile_yaml(
        """
defaults: {}
aliases:
  - name: reference-parent
    candidates:
      - alias_reference: reference-branch
        priority: 50
  - name: reference-branch
    candidates:
      - provider: openrouter
        model: openrouter/reference-target
        route_family: codex_openrouter_completion_adapter
        anthropic_route_family: anthropic_openrouter_completion_adapter
        priority: 50
"""
    )
    previous = snapshot_select.get_active_routing_snapshot()
    snapshot_select.set_active_routing_snapshot(snapshot)
    try:
        direct_alias = snapshot_select._lookup_active_snapshot_canonical_alias(
            "REFERENCE-BRANCH"
        )
        parent_alias = snapshot_select._lookup_active_snapshot_canonical_alias(
            "reference-parent"
        )
        assert direct_alias == "reference-branch"
        assert parent_alias == "reference-parent"

        direct_codex = snapshot_select._select_snapshot_candidates(
            direct_alias,
            ingress="codex",
        )
        referenced_codex = snapshot_select._select_snapshot_candidates(
            parent_alias,
            ingress="codex",
        )
        direct_anthropic = snapshot_select._select_snapshot_candidates(
            direct_alias,
            ingress="anthropic",
        )
        referenced_anthropic = snapshot_select._select_snapshot_candidates(
            parent_alias,
            ingress="anthropic",
        )

        def identities(
            candidates: tuple[dict[str, object], ...],
        ) -> list[tuple[object, object, object]]:
            return [
                (
                    candidate["provider"],
                    candidate["model"],
                    candidate["route_family"],
                )
                for candidate in candidates
            ]

        assert identities(referenced_codex) == identities(direct_codex)
        assert identities(referenced_anthropic) == identities(direct_anthropic)
    finally:
        snapshot_select.set_active_routing_snapshot(previous)


def test_daily_alias_reference_schedule_is_half_open_and_wraps_midnight() -> None:
    snapshot = compiler.compile_yaml(
        """
defaults: {}
aliases:
  - name: parent
    candidates:
      - alias_reference: night
        priority: 110
        schedule:
          start_time: "22:00:00"
          end_time: "08:00:00"
          utc_offset: "+08:00"
      - alias_reference: always
        priority: 100
  - name: night
    candidates:
      - provider: alibaba_token_plan
        model: alibaba_token_plan/deepseek-v4-pro
        route_family: codex_alibaba_token_plan_chat_completions_adapter
        priority: 100
  - name: always
    candidates:
      - provider: kimi_code
        model: kimi_code/k3
        route_family: codex_kimi_chat_completions_adapter
        priority: 100
"""
    )
    previous = snapshot_select.get_active_routing_snapshot()
    snapshot_select.set_active_routing_snapshot(snapshot)
    try:
        window_start = dt.datetime(2026, 8, 17, 14, 0, tzinfo=dt.timezone.utc)
        just_before_end = dt.datetime(2026, 8, 17, 23, 59, 59, tzinfo=dt.timezone.utc)
        window_end = dt.datetime(2026, 8, 18, 0, 0, tzinfo=dt.timezone.utc)

        def models(now: dt.datetime) -> list[str]:
            return [
                str(candidate["model"])
                for candidate in snapshot_select._select_snapshot_candidates(
                    "parent",
                    ingress="codex",
                    now_utc=now,
                )
            ]

        assert models(window_start) == [
            "alibaba_token_plan/deepseek-v4-pro",
            "kimi_code/k3",
        ]
        assert models(just_before_end) == [
            "alibaba_token_plan/deepseek-v4-pro",
            "kimi_code/k3",
        ]
        assert models(window_end) == ["kimi_code/k3"]
        preserved = [
            str(candidate["model"])
            for candidate in snapshot_select._select_snapshot_candidates(
                "parent",
                ingress="codex",
                now_utc=window_end,
                include_out_of_schedule=True,
            )
        ]
        assert preserved == [
            "alibaba_token_plan/deepseek-v4-pro",
            "kimi_code/k3",
        ]
    finally:
        snapshot_select.set_active_routing_snapshot(previous)


def test_same_day_daily_schedule_is_half_open() -> None:
    snapshot = compiler.compile_yaml(
        """
defaults: {}
aliases:
  - name: scheduled
    candidates:
      - provider: alibaba_token_plan
        model: alibaba_token_plan/qwen3.8-max
        route_family: codex_alibaba_token_plan_chat_completions_adapter
        priority: 110
        schedule:
          start_time: "09:00:00"
          end_time: "17:00:00"
          utc_offset: "+00:00"
      - provider: openai
        model: gpt-5.6-terra
        route_family: codex_responses
        priority: 0
"""
    )
    previous = snapshot_select.get_active_routing_snapshot()
    snapshot_select.set_active_routing_snapshot(snapshot)
    try:
        window_start = dt.datetime(2026, 8, 18, 9, 0, tzinfo=dt.timezone.utc)
        just_before_end = dt.datetime(2026, 8, 18, 16, 59, 59, tzinfo=dt.timezone.utc)
        window_end = dt.datetime(2026, 8, 18, 17, 0, tzinfo=dt.timezone.utc)

        def models(now: dt.datetime) -> list[str]:
            return [
                str(candidate["model"])
                for candidate in snapshot_select._select_snapshot_candidates(
                    "scheduled",
                    ingress="codex",
                    now_utc=now,
                )
            ]

        assert models(window_start) == [
            "alibaba_token_plan/qwen3.8-max",
            "gpt-5.6-terra",
        ]
        assert models(just_before_end) == [
            "alibaba_token_plan/qwen3.8-max",
            "gpt-5.6-terra",
        ]
        assert models(window_end) == ["gpt-5.6-terra"]
    finally:
        snapshot_select.set_active_routing_snapshot(previous)


def test_canonical_work_other_promotes_deepseek_only_inside_daily_window() -> None:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
        DEFAULT_CONFIG_DIR,
        compile_directory,
    )

    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    previous = snapshot_select.get_active_routing_snapshot()
    snapshot_select.set_active_routing_snapshot(snapshot)
    try:
        inside = snapshot_select._select_snapshot_candidates(
            "work-other",
            ingress="codex",
            now_utc=dt.datetime(2026, 8, 18, 15, 0, tzinfo=dt.timezone.utc),
        )
        outside = snapshot_select._select_snapshot_candidates(
            "work-other",
            ingress="codex",
            now_utc=dt.datetime(2026, 8, 18, 1, 0, tzinfo=dt.timezone.utc),
        )
        assert [candidate["model"] for candidate in inside] == [
            "alibaba_token_plan/deepseek-v4-pro",
            "kimi_code/k3",
            "oa_xai/grok-4.6",
            "cursor_agent/cursor-grok-4.6-high",
        ]
        assert [candidate["model"] for candidate in outside] == [
            "kimi_code/k3",
            "oa_xai/grok-4.6",
            "cursor_agent/cursor-grok-4.6-high",
        ]
        assert all(
            "qwen" not in str(candidate["model"])
            for candidate in (*inside, *outside)
        )

        inside_anthropic = snapshot_select._select_snapshot_candidates(
            "work-other",
            ingress="anthropic",
            now_utc=dt.datetime(2026, 8, 18, 15, 0, tzinfo=dt.timezone.utc),
        )
        outside_anthropic = snapshot_select._select_snapshot_candidates(
            "work-other",
            ingress="anthropic",
            now_utc=dt.datetime(2026, 8, 18, 1, 0, tzinfo=dt.timezone.utc),
        )
        assert [candidate["model"] for candidate in inside_anthropic] == [
            "alibaba_token_plan/deepseek-v4-pro",
            "kimi_code/k3",
            "oa_xai/grok-4.6",
            "cursor_agent/cursor-grok-4.6-high",
        ]
        assert [candidate["model"] for candidate in outside_anthropic] == [
            "kimi_code/k3",
            "oa_xai/grok-4.6",
            "cursor_agent/cursor-grok-4.6-high",
        ]
        assert (
            inside_anthropic[0]["route_family"]
            == "anthropic_alibaba_token_plan_chat_completions_adapter"
        )
        preserved = snapshot_select._select_snapshot_candidates(
            "work-other",
            ingress="codex",
            now_utc=dt.datetime(2026, 8, 18, 1, 0, tzinfo=dt.timezone.utc),
            include_out_of_schedule=True,
        )
        assert [candidate["model"] for candidate in preserved] == [
            "alibaba_token_plan/deepseek-v4-pro",
            "kimi_code/k3",
            "oa_xai/grok-4.6",
            "cursor_agent/cursor-grok-4.6-high",
        ]
    finally:
        snapshot_select.set_active_routing_snapshot(previous)
