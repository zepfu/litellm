"""RED-phase tests for Wave 3: compiler -> immutable RoutingSnapshot.

Modules under test do not exist yet:
``config_compiler.py``, ``config_snapshot.py`` (and ``config_schema.py``).
"""

from __future__ import annotations

import dataclasses
import itertools

import pytest
from pydantic import ValidationError

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (  # type: ignore[import-not-found]
    config_compiler as compiler,
)


_RAW_YAML = """
defaults:
  route_family: codex_responses
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: openrouter/cohere/north-mini-code:free
        route_family: codex_openrouter_completion_adapter
        priority: 100
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 0
"""


def test_compile_produces_snapshot_with_epoch_hash_version() -> None:
    """Compiling valid YAML produces a snapshot carrying config_epoch/hash/version."""
    snapshot = compiler.compile_yaml(_RAW_YAML)
    assert isinstance(snapshot.config_epoch, int)
    assert isinstance(snapshot.config_hash, str) and snapshot.config_hash
    assert isinstance(snapshot.config_version, str) and snapshot.config_version


def test_snapshot_is_immutable() -> None:
    """Compiled snapshot is frozen — mutation raises."""
    snapshot = compiler.compile_yaml(_RAW_YAML)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        snapshot.config_epoch = 999  # type: ignore[misc]


def test_rejects_unknown_keys_and_malformed_at_compile() -> None:
    """Malformed YAML fails compile with a validation error, not a silent partial compile."""
    malformed = _RAW_YAML + "\n  unknown_top_level_key: true\n"
    with pytest.raises((ValidationError, compiler.ConfigCompileError)):
        compiler.compile_yaml(malformed)


def test_priority_descending_with_zero_last_resort_in_snapshot() -> None:
    """Snapshot candidate ordering is descending; priority 0 is placed last."""
    snapshot = compiler.compile_yaml(_RAW_YAML)
    basic_alias = snapshot.aliases["basic"]
    models_in_order = [c.model for c in basic_alias.candidates]
    assert models_in_order == [
        "openrouter/cohere/north-mini-code:free",
        "gpt-5.4-mini",
    ]
    assert basic_alias.candidates[-1].priority == 0


def test_canonical_sota_zai_compiles_coding_plan_ahead_of_alibaba() -> None:
    """Public sota-zai prefers Coding Plan glm-5.3 (110) then Alibaba glm-5.2 (100)."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
        DEFAULT_CONFIG_DIR,
        compile_directory,
    )

    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    alias = snapshot.aliases["sota-zai"]
    identities = [
        (
            candidate.provider,
            candidate.model,
            candidate.route_family,
            candidate.priority,
        )
        for candidate in alias.candidates
    ]
    assert identities == [
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
    ]
    coding_plan, alibaba = alias.candidates
    assert coding_plan.priority > alibaba.priority
    assert coding_plan.anthropic_route_family is None
    assert (
        alibaba.anthropic_route_family
        == "anthropic_alibaba_token_plan_chat_completions_adapter"
    )
    assert coding_plan.reasoning_effort == "max"
    assert alibaba.reasoning_effort == "max"
    assert "aawm-sota-zai" not in snapshot.aliases
    assert "sota-zcode" not in snapshot.aliases
    sota = snapshot.aliases["sota"]
    assert sota.dispatch is not None
    assert all(rule.target_alias != "sota-zai" for rule in sota.dispatch.by_tui)
    assert sota.dispatch.default != "sota-zai"


def test_sota_zai_coding_plan_candidate_rejects_anthropic_route_family() -> None:
    raw = """
defaults: {}
aliases:
  - name: sota-zai
    candidates:
      - provider: zai_coding_plan
        model: zai_coding_plan/glm-5.3
        route_family: codex_zai_coding_plan_chat_completions_adapter
        anthropic_route_family: anthropic_alibaba_token_plan_chat_completions_adapter
        priority: 110
"""
    with pytest.raises(compiler.ConfigCompileError, match="Codex-only"):
        compiler.compile_yaml(raw)


def test_proportional_weights_normalized_in_snapshot() -> None:
    """Compiler normalizes proportional weights into the snapshot."""
    raw = """
defaults: {}
aliases:
  - name: basic
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
    weights = {c.model: c.weight for c in snapshot.aliases["basic"].candidates}
    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert pytest.approx(weights["a"], rel=1e-6) == 0.25
    assert pytest.approx(weights["b"], rel=1e-6) == 0.75


def test_tui_attached_flag_compiles_into_snapshot() -> None:
    """A candidate's tui_attached flag is preserved through compilation."""
    raw = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 10
        tui_attached: Claude
"""
    snapshot = compiler.compile_yaml(raw)
    candidate = snapshot.aliases["basic"].candidates[0]
    assert candidate.tui_attached == "Claude"


def test_tui_excluded_flag_compiles_into_snapshot_and_hash() -> None:
    """CFG-008: tui_excluded survives compilation and feeds the semantic hash."""
    base_raw = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openai
        model: gpt-5.6-luna
        route_family: codex_responses
        priority: 0
"""
    with_excluded_raw = base_raw.replace(
        "priority: 0", "priority: 0\n        tui_excluded: Claude", 1
    )

    snapshot_plain = compiler.compile_yaml(base_raw)
    snapshot_with = compiler.compile_yaml(with_excluded_raw)

    assert snapshot_plain.aliases["basic"].candidates[0].tui_excluded is None
    assert snapshot_with.aliases["basic"].candidates[0].tui_excluded == "Claude"
    # The exclusion participates in semantic config identity.
    assert snapshot_plain.config_hash != snapshot_with.config_hash


def test_schedule_windows_utc_only_in_snapshot() -> None:
    """Compiled snapshot preserves UTC schedule windows; overlaps resolve deterministically."""
    raw = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: alibaba_token_plan
        model: alibaba_token_plan/qwen3.8-max-preview
        route_family: codex_alibaba_token_plan_chat_completions_adapter
        priority: 500
        schedule:
          start: "2026-07-01T00:00:00Z"
          end: "2026-07-15T00:00:00Z"
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 0
"""
    snapshot = compiler.compile_yaml(raw)
    promo_candidate = snapshot.aliases["basic"].candidates[0]
    assert promo_candidate.schedule is not None
    assert promo_candidate.schedule.start.utcoffset().total_seconds() == 0
    assert promo_candidate.schedule.kind == "absolute"


def test_daily_schedule_windows_compile_and_change_config_hash() -> None:
    absolute = """
defaults: {}
aliases:
  - name: scheduled
    candidates:
      - provider: alibaba_token_plan
        model: alibaba_token_plan/qwen3.8-max
        route_family: codex_alibaba_token_plan_chat_completions_adapter
        priority: 100
        schedule:
          start: "2026-07-01T00:00:00Z"
          end: "2026-07-15T00:00:00Z"
"""
    daily = """
defaults: {}
aliases:
  - name: scheduled
    candidates:
      - provider: alibaba_token_plan
        model: alibaba_token_plan/qwen3.8-max
        route_family: codex_alibaba_token_plan_chat_completions_adapter
        priority: 100
        schedule:
          start_time: "22:00:00"
          end_time: "08:00:00"
          utc_offset: "+08:00"
"""
    snapshot_absolute = compiler.compile_yaml(absolute)
    snapshot_daily = compiler.compile_yaml(daily)
    daily_candidate = snapshot_daily.aliases["scheduled"].candidates[0]
    assert daily_candidate.schedule is not None
    assert daily_candidate.schedule.kind == "daily"
    assert snapshot_absolute.config_hash != snapshot_daily.config_hash


def test_named_daily_schedule_compiles_with_iana_timezone() -> None:
    raw = """
defaults: {}
aliases:
  - name: scheduled
    candidates:
      - provider: openrouter
        model: openrouter/scheduled
        route_family: codex_openrouter_completion_adapter
        priority: 100
        schedule:
          start_time: "03:00:00"
          end_time: "23:00:00"
          timezone: "America/Los_Angeles"
"""
    snapshot = compiler.compile_yaml(raw)
    schedule = snapshot.aliases["scheduled"].candidates[0].schedule

    assert schedule is not None
    assert schedule.kind == "daily"
    assert schedule.start_time.isoformat() == "03:00:00"
    assert schedule.end_time.isoformat() == "23:00:00"
    assert schedule.utc_offset is None
    assert schedule.timezone == "America/Los_Angeles"


def test_canonical_zai_candidates_use_dst_aware_off_peak_window() -> None:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
        DEFAULT_CONFIG_DIR,
        compile_directory,
    )

    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    restricted_aliases = (
        "provider-zai_coding_plan",
        "basic-other",
        "work-other",
        "auto-review-other",
    )

    for alias_name in restricted_aliases:
        candidate = next(
            candidate
            for candidate in snapshot.aliases[alias_name].candidates
            if getattr(candidate, "provider", None) == "zai_coding_plan"
        )
        assert candidate.schedule is not None
        assert candidate.schedule.kind == "daily"
        assert candidate.schedule.start_time.isoformat() == "03:00:00"
        assert candidate.schedule.end_time.isoformat() == "23:00:00"
        assert candidate.schedule.timezone == "America/Los_Angeles"
        assert candidate.schedule.utc_offset is None

    sota_zai_candidate = snapshot.aliases["sota-zai"].candidates[0]
    assert sota_zai_candidate.provider == "zai_coding_plan"
    assert sota_zai_candidate.schedule is None


def test_alias_reference_schedule_participates_in_config_hash() -> None:
    """CFG-020: alias_reference schedules feed the semantic digest."""
    unscheduled = """
defaults: {}
aliases:
  - name: parent
    candidates:
      - alias_reference: child
        priority: 110
  - name: child
    candidates:
      - provider: alibaba_token_plan
        model: alibaba_token_plan/deepseek-v4-pro
        route_family: codex_alibaba_token_plan_chat_completions_adapter
        priority: 100
"""
    scheduled = """
defaults: {}
aliases:
  - name: parent
    candidates:
      - alias_reference: child
        priority: 110
        schedule:
          start_time: "22:00:00"
          end_time: "08:00:00"
          utc_offset: "+08:00"
  - name: child
    candidates:
      - provider: alibaba_token_plan
        model: alibaba_token_plan/deepseek-v4-pro
        route_family: codex_alibaba_token_plan_chat_completions_adapter
        priority: 100
"""
    equivalent = """
defaults: {}
aliases:
  - name: parent
    candidates:
      - alias_reference: child
        priority: 110
        schedule:
          start_time: "22:00"
          end_time: "08:00"
          utc_offset: "UTC+8"
  - name: child
    candidates:
      - provider: alibaba_token_plan
        model: alibaba_token_plan/deepseek-v4-pro
        route_family: codex_alibaba_token_plan_chat_completions_adapter
        priority: 100
"""
    snapshot_plain = compiler.compile_yaml(unscheduled)
    snapshot_scheduled = compiler.compile_yaml(scheduled)
    snapshot_equivalent = compiler.compile_yaml(equivalent)
    reference = snapshot_scheduled.aliases["parent"].candidates[0]
    assert reference.schedule is not None
    assert reference.schedule.kind == "daily"
    assert snapshot_plain.config_hash != snapshot_scheduled.config_hash
    assert snapshot_equivalent.config_hash == snapshot_scheduled.config_hash


def test_canonical_work_other_compiles_scheduled_deepseek_without_alibaba() -> None:
    """CFG-041: work-other orders scheduled DeepSeek, Z.AI, Moonshot, then xAI."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_snapshot import (
        AliasReference,
        RoutingCandidate,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
        DEFAULT_CONFIG_DIR,
        compile_directory,
    )

    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    alias = snapshot.aliases["work-other"]
    identities = [
        (
            ("REF", entry.alias_name, None, entry.priority)
            if isinstance(entry, AliasReference)
            else (entry.provider, entry.model, entry.route_family, entry.priority)
        )
        for entry in alias.candidates
    ]
    assert identities == [
        ("REF", "sota-deepseek", None, 110),
        (
            "zai_coding_plan",
            "zai_coding_plan/glm-5.3-flash",
            "codex_zai_coding_plan_chat_completions_adapter",
            100,
        ),
        ("REF", "sota-moonshot", None, 90),
        ("REF", "sota-xai", None, 80),
    ]
    deepseek = alias.candidates[0]
    assert isinstance(deepseek, AliasReference)
    assert deepseek.schedule is not None
    assert deepseek.schedule.kind == "daily"
    assert isinstance(alias.candidates[1], RoutingCandidate)
    owner = snapshot.aliases["sota-deepseek"].candidates[0]
    assert owner.schedule is None


def test_canonical_work_compiles_current_graph() -> None:
    """CFG-041: work delegates first to work-other, then keeps native tails."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_snapshot import (
        AliasReference,
        RoutingCandidate,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
        DEFAULT_CONFIG_DIR,
        compile_directory,
    )

    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    entries = snapshot.aliases["work"].candidates

    assert len(entries) == 4
    assert isinstance(entries[0], AliasReference)
    assert all(
        isinstance(entry, RoutingCandidate)
        for index, entry in enumerate(entries)
        if index != 0
    )
    assert [
        (
            ("REF", entry.alias_name, None, entry.priority)
            if isinstance(entry, AliasReference)
            else (entry.provider, entry.model, entry.route_family, entry.priority)
        )
        for entry in entries
    ] == [
        ("REF", "work-other", None, 110),
        ("anthropic", "claude-sonnet-5[1m]", "anthropic_messages", 80),
        ("anthropic", "claude-sonnet-5", "anthropic_messages", 70),
        ("openai", "gpt-5.6-luna", "codex_responses", 0),
    ]
    sonnet_1m, sonnet = entries[1:3]
    assert isinstance(sonnet_1m, RoutingCandidate)
    assert isinstance(sonnet, RoutingCandidate)
    for candidate in (sonnet_1m, sonnet):
        assert candidate.anthropic_route_family == "anthropic_messages"
        assert candidate.reasoning_effort == "max"
        assert candidate.tui_attached == "Claude"
    luna = entries[-1]
    assert isinstance(luna, RoutingCandidate)
    assert luna.reasoning_effort == "max"

    direct_models = {
        entry.model for entry in entries if isinstance(entry, RoutingCandidate)
    }
    assert direct_models.isdisjoint(
        {
            "zai_coding_plan/glm-5.3-flash",
            "cursor_agent/cursor-grok-4.6-high",
            "xai/grok-4.6",
            "oa_xai/grok-4.6",
        }
    )


def test_canonical_provider_openai_compiles_cheapest_first_at_low_effort() -> None:
    """CFG-041: provider-openai is Luna, Terra, Sol with low effort."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
        DEFAULT_CONFIG_DIR,
        compile_directory,
    )

    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    alias = snapshot.aliases["provider-openai"]
    assert [
        (
            candidate.provider,
            candidate.model,
            candidate.route_family,
            candidate.priority,
            candidate.reasoning_effort,
        )
        for candidate in alias.candidates
    ] == [
        ("openai", "gpt-5.6-luna", "codex_responses", 100, "low"),
        ("openai", "gpt-5.6-terra", "codex_responses", 90, "low"),
        ("openai", "gpt-5.6-sol", "codex_responses", 0, "low"),
    ]


def test_canonical_provider_nvidia_compiles_closed_nim_set_without_alias_reference() -> None:
    """Live provider-nvidia is five NVIDIA NIM models on NVIDIA-credential families."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_snapshot import (
        AliasReference,
        RoutingCandidate,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
        DEFAULT_CONFIG_DIR,
        compile_directory,
    )

    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    alias = snapshot.aliases["provider-nvidia"]
    assert alias.dispatch is None
    assert all(isinstance(entry, RoutingCandidate) for entry in alias.candidates)
    assert all(not isinstance(entry, AliasReference) for entry in alias.candidates)
    identities = [
        (entry.provider, entry.model, entry.route_family, entry.priority)
        for entry in alias.candidates
    ]
    assert identities == [
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
    ]
    for entry in alias.candidates:
        assert entry.anthropic_route_family is None


def test_nvidia_credential_domain_rejects_foreign_provider_and_non_nim_family() -> None:
    """NVIDIA-credential families stay NVIDIA-native; NVIDIA models stay on NIM families."""
    foreign_provider = """
defaults: {}
aliases:
  - name: mixed
    candidates:
      - provider: openrouter
        model: nvidia/deepseek-ai/deepseek-v3.2
        route_family: codex_nvidia_completion_adapter
        priority: 100
"""
    with pytest.raises(compiler.ConfigCompileError, match="NVIDIA-credential"):
        compiler.compile_yaml(foreign_provider)

    nvidia_on_codex = """
defaults: {}
aliases:
  - name: mixed
    candidates:
      - provider: nvidia
        model: nvidia/deepseek-ai/deepseek-v3.2
        route_family: codex_responses
        priority: 100
"""
    with pytest.raises(compiler.ConfigCompileError, match="NVIDIA-native"):
        compiler.compile_yaml(nvidia_on_codex)

    nvidia_on_openrouter = """
defaults: {}
aliases:
  - name: mixed
    candidates:
      - provider: nvidia
        model: nvidia/deepseek-ai/deepseek-v3.2
        route_family: codex_openrouter_completion_adapter
        priority: 100
"""
    with pytest.raises(compiler.ConfigCompileError, match="NVIDIA-native"):
        compiler.compile_yaml(nvidia_on_openrouter)


def test_inheritance_resolves_at_compile() -> None:
    """Typed inheritance (defaults -> alias -> candidate) compiles without ambiguity."""
    snapshot = compiler.compile_yaml(_RAW_YAML)
    basic_alias = snapshot.aliases["basic"]
    or_candidate = next(c for c in basic_alias.candidates if c.provider == "openrouter")
    assert or_candidate.route_family == "codex_openrouter_completion_adapter"
    fallback_candidate = next(c for c in basic_alias.candidates if c.provider == "openai")
    assert fallback_candidate.route_family == "codex_responses"


def test_error_class_refs_open_vocabulary_at_compile() -> None:
    """Error rules may reference class names not in the seed registry; compile does not fail."""
    raw = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 0
        error_rules:
          - class_name: a_totally_new_future_class
            cools: true
"""
    snapshot = compiler.compile_yaml(raw)
    candidate = snapshot.aliases["basic"].candidates[0]
    assert candidate.error_rules[0].class_name == "a_totally_new_future_class"


def test_rejects_arbitrary_behavior_at_compile() -> None:
    """A candidate referencing an unregistered code behavior is rejected at compile time."""
    raw = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: totally_unregistered_provider_xyz
        model: whatever
        route_family: codex_responses
        priority: 0
"""
    with pytest.raises((ValidationError, compiler.ConfigCompileError)):
        compiler.compile_yaml(raw)


def test_reasoning_effort_snapshot_field_and_semantic_hash() -> None:
    """CFG-006: optional reasoning_effort survives compile and feeds the hash."""
    base_raw = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 0
"""
    with_effort_raw = base_raw.replace(
        "priority: 0", "priority: 0\n        reasoning_effort: low", 1
    )

    snapshot_plain = compiler.compile_yaml(base_raw)
    snapshot_with = compiler.compile_yaml(with_effort_raw)

    assert snapshot_plain.aliases["basic"].candidates[0].reasoning_effort is None
    assert snapshot_with.aliases["basic"].candidates[0].reasoning_effort == "low"
    # The configured value participates in semantic config identity.
    assert snapshot_plain.config_hash != snapshot_with.config_hash


def test_reasoning_effort_malformed_fails_compile_closed() -> None:
    """CFG-006: a malformed reasoning_effort rejects the whole compile."""
    raw = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 0
        reasoning_effort: extreme
"""
    with pytest.raises((ValidationError, compiler.ConfigCompileError)):
        compiler.compile_yaml(raw)


def test_alias_multi_agent_version_compiles_into_snapshot_and_hash() -> None:
    base_raw = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 0
"""
    with_version_raw = base_raw.replace(
        "name: basic", "name: basic\n    multi_agent_version: v2", 1
    )

    snapshot_plain = compiler.compile_yaml(base_raw)
    snapshot_with = compiler.compile_yaml(with_version_raw)

    assert snapshot_plain.aliases["basic"].multi_agent_version is None
    assert snapshot_with.aliases["basic"].multi_agent_version == "v2"
    assert snapshot_plain.config_hash != snapshot_with.config_hash


# ===========================================================================
# Wave 3: R3-4 -- semantic digest stability across processes and formatting
# ===========================================================================


def test_semantic_key_tag_is_stable_across_processes_and_source_formatting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3-4 RED pre-fix: recompile semantically identical YAML after resetting
    the compiler counter (simulating restart) and after comment/whitespace/
    key-order edits; assert identical full semantic digest while config_epoch
    differs and source_hash may differ. A priority/model/route-family change
    must change the semantic digest.

    Pre-fix failure: config_hash = sha256(raw_yaml), so any formatting change
    (comment, whitespace, key order) produces a different hash.
    """
    yaml_original = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: openrouter/cohere/north-mini-code:free
        route_family: codex_openrouter_completion_adapter
        priority: 100
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 0
"""

    # Semantically identical: comment added, mapping-key order changed, extra whitespace
    yaml_reformatted = """
# Reformatted: comment added, key order changed, extra whitespace
defaults: {}
aliases:
  - name: basic
    candidates:
      - route_family: codex_openrouter_completion_adapter
        provider: openrouter
        priority: 100
        model: openrouter/cohere/north-mini-code:free
      - route_family: codex_responses
        provider:   openai
        priority:   0
        model:   gpt-5.4-mini
"""

    snapshot_a = compiler.compile_yaml(yaml_original)
    snapshot_b = compiler.compile_yaml(yaml_reformatted)

    # config_epoch must differ (monotonic counter)
    assert snapshot_b.config_epoch != snapshot_a.config_epoch, (
        "config_epoch must be monotonically increasing across compiles"
    )

    # Semantic digest (config_hash) must be IDENTICAL for semantically
    # identical YAML regardless of source formatting
    assert snapshot_b.config_hash == snapshot_a.config_hash, (
        "R3-4: semantically identical YAML (comment/whitespace/key-order "
        "changes only) must produce the same config_hash (semantic digest). "
        f"Pre-fix: config_hash = sha256(raw_yaml) so formatting changes "
        f"produce different hashes. Got {snapshot_a.config_hash!r} vs "
        f"{snapshot_b.config_hash!r}"
    )

    # Simulate a fresh process where the telemetry counter restarts. The
    # semantic digest must remain stable even when the integer epoch changes
    # or collides with a prior process's value.
    monkeypatch.setattr(compiler, "_epoch_counter", itertools.count(1))
    snapshot_after_restart = compiler.compile_yaml(yaml_reformatted)
    assert snapshot_after_restart.config_epoch != snapshot_b.config_epoch
    assert snapshot_after_restart.config_hash == snapshot_a.config_hash

    source_hash_a = getattr(snapshot_a, "source_hash", None)
    source_hash_b = getattr(snapshot_b, "source_hash", None)
    if source_hash_a is not None and source_hash_b is not None:
        assert source_hash_a != source_hash_b

    # A priority change must change the semantic digest
    yaml_priority_changed = yaml_original.replace("priority: 100", "priority: 200")
    snapshot_c = compiler.compile_yaml(yaml_priority_changed)
    assert snapshot_c.config_hash != snapshot_a.config_hash, (
        "a priority change must produce a different semantic digest"
    )

    # A model change must change the semantic digest
    yaml_model_changed = yaml_original.replace("gpt-5.4-mini", "gpt-5.4-max")
    snapshot_d = compiler.compile_yaml(yaml_model_changed)
    assert snapshot_d.config_hash != snapshot_a.config_hash, (
        "a model change must produce a different semantic digest"
    )

    # A route_family change must change the semantic digest
    yaml_rf_changed = yaml_original.replace(
        "codex_openrouter_completion_adapter", "codex_responses"
    )
    snapshot_e = compiler.compile_yaml(yaml_rf_changed)
    assert snapshot_e.config_hash != snapshot_a.config_hash, (
        "a route_family change must produce a different semantic digest"
    )


def test_auto_review_yaml_compiles_combined_public_alias_graph() -> None:
    """CFG-041: one document owns both public auto-review alias names."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_snapshot import (
        AliasReference,
        RoutingCandidate,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
        DEFAULT_CONFIG_DIR,
        compile_directory,
    )

    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    auto_review = snapshot.aliases["auto-review"]
    codex_auto_review = snapshot.aliases["codex-auto-review"]
    assert auto_review.dispatch is None
    assert codex_auto_review.dispatch is None
    assert len(codex_auto_review.candidates) == 1
    public_reference = codex_auto_review.candidates[0]
    assert isinstance(public_reference, AliasReference)
    assert (public_reference.alias_name, public_reference.priority) == (
        "auto-review",
        100,
    )

    helper_reference = auto_review.candidates[0]
    assert isinstance(helper_reference, AliasReference)
    assert (helper_reference.alias_name, helper_reference.priority) == (
        "auto-review-other",
        100,
    )
    concrete = auto_review.candidates[1:]
    assert all(isinstance(candidate, RoutingCandidate) for candidate in concrete)
    assert [
        (
            candidate.provider,
            candidate.model,
            candidate.route_family,
            candidate.priority,
            candidate.reasoning_effort,
        )
        for candidate in concrete
    ] == [
        (
            "openai",
            "gpt-5.6-luna",
            "codex_responses",
            90,
            "low",
        ),
        (
            "openrouter",
            "openrouter/~deepseek/deepseek-v4-flash-latest",
            "codex_openrouter_completion_adapter",
            0,
            "low",
        ),
    ]

    invalid = """
defaults: {}
aliases:
  - name: codex-auto-review
    candidates:
      - provider: alibaba_token_plan
        model: alibaba_token_plan/deepseek-v4-flash-0731
        route_family: codex_alibaba_token_plan_chat_completions_adapter
        priority: 100
        unknown_key: true
"""
    with pytest.raises((ValidationError, compiler.ConfigCompileError)):
        compiler.compile_yaml(invalid)


def test_auto_review_other_preserves_low_effort_helper_candidates() -> None:
    """CFG-041: every concrete helper candidate keeps low reasoning."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
        DEFAULT_CONFIG_DIR,
        compile_directory,
    )

    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    helper = snapshot.aliases["auto-review-other"]
    assert [
        (
            candidate.model,
            candidate.priority,
            candidate.reasoning_effort,
            candidate.schedule.kind if candidate.schedule is not None else None,
        )
        for candidate in helper.candidates
    ] == [
        ("alibaba_token_plan/deepseek-v4-flash-0731", 100, "low", "daily"),
        ("zai_coding_plan/glm-5.3-flash", 90, "low", "daily"),
        ("cursor_agent/composer-2.5", 80, "low", None),
    ]
