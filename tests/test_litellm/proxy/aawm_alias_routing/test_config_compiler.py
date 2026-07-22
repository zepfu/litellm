"""RED-phase tests for Wave 3: compiler -> immutable RoutingSnapshot.

Modules under test do not exist yet:
``config_compiler.py``, ``config_snapshot.py`` (and ``config_schema.py``).
"""

from __future__ import annotations

import dataclasses

import pytest
from pydantic import ValidationError

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (  # type: ignore[import-not-found]
    config_compiler as compiler,
)


_RAW_YAML = """
defaults:
  route_family: codex_responses
aliases:
  - name: read
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
    read_alias = snapshot.aliases["read"]
    models_in_order = [c.model for c in read_alias.candidates]
    assert models_in_order == [
        "openrouter/cohere/north-mini-code:free",
        "gpt-5.4-mini",
    ]
    assert read_alias.candidates[-1].priority == 0


def test_proportional_weights_normalized_in_snapshot() -> None:
    """Compiler normalizes proportional weights into the snapshot."""
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
    weights = {c.model: c.weight for c in snapshot.aliases["read"].candidates}
    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert pytest.approx(weights["a"], rel=1e-6) == 0.25
    assert pytest.approx(weights["b"], rel=1e-6) == 0.75


def test_tui_attached_flag_compiles_into_snapshot() -> None:
    """A candidate's tui_attached flag is preserved through compilation."""
    raw = """
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 10
        tui_attached: Claude
"""
    snapshot = compiler.compile_yaml(raw)
    candidate = snapshot.aliases["read"].candidates[0]
    assert candidate.tui_attached == "Claude"


def test_schedule_windows_utc_only_in_snapshot() -> None:
    """Compiled snapshot preserves UTC schedule windows; overlaps resolve deterministically."""
    raw = """
defaults: {}
aliases:
  - name: read
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
    promo_candidate = snapshot.aliases["read"].candidates[0]
    assert promo_candidate.schedule is not None
    assert promo_candidate.schedule.start.utcoffset().total_seconds() == 0


def test_inheritance_resolves_at_compile() -> None:
    """Typed inheritance (defaults -> alias -> candidate) compiles without ambiguity."""
    snapshot = compiler.compile_yaml(_RAW_YAML)
    read_alias = snapshot.aliases["read"]
    or_candidate = next(c for c in read_alias.candidates if c.provider == "openrouter")
    assert or_candidate.route_family == "codex_openrouter_completion_adapter"
    fallback_candidate = next(c for c in read_alias.candidates if c.provider == "openai")
    assert fallback_candidate.route_family == "codex_responses"


def test_error_class_refs_open_vocabulary_at_compile() -> None:
    """Error rules may reference class names not in the seed registry; compile does not fail."""
    raw = """
defaults: {}
aliases:
  - name: read
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
    candidate = snapshot.aliases["read"].candidates[0]
    assert candidate.error_rules[0].class_name == "a_totally_new_future_class"


def test_rejects_arbitrary_behavior_at_compile() -> None:
    """A candidate referencing an unregistered code behavior is rejected at compile time."""
    raw = """
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: totally_unregistered_provider_xyz
        model: whatever
        route_family: codex_responses
        priority: 0
"""
    with pytest.raises((ValidationError, compiler.ConfigCompileError)):
        compiler.compile_yaml(raw)
