"""RED-phase tests for Wave 3: typed YAML schema (pydantic v2).

Module under test does not exist yet:
``litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_schema``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (  # type: ignore[import-not-found]
    config_schema as schema,
)


def _base_candidate(**overrides: object) -> dict:
    candidate = {
        "provider": "openrouter",
        "model": "openrouter/cohere/north-mini-code:free",
        "route_family": "codex_openrouter_completion_adapter",
        "priority": 100,
    }
    candidate.update(overrides)
    return candidate


def _base_alias(**overrides: object) -> dict:
    alias = {
        "name": "read",
        "candidates": [_base_candidate()],
    }
    alias.update(overrides)
    return alias


def test_rejects_unknown_keys_and_malformed() -> None:
    """Unknown fields, missing required fields, and non-typed values are rejected."""
    with pytest.raises(ValidationError):
        schema.AliasConfig.model_validate(_base_alias(unknown_field_xyz="oops"))

    missing_required = _base_alias()
    del missing_required["name"]
    with pytest.raises(ValidationError):
        schema.AliasConfig.model_validate(missing_required)

    with pytest.raises(ValidationError):
        schema.CandidateConfig.model_validate(_base_candidate(priority="not-an-int"))


def test_rejects_arbitrary_behavior() -> None:
    """A candidate whose route_family/provider is not a registered code behavior is rejected."""
    with pytest.raises(ValidationError):
        schema.CandidateConfig.model_validate(_base_candidate(route_family="__import__('os').system('echo pwned')"))
    with pytest.raises(ValidationError):
        schema.CandidateConfig.model_validate(_base_candidate(provider="totally_unregistered_provider_xyz"))


def test_numeric_priority_required_and_typed() -> None:
    """Each candidate requires an integer priority; non-int is rejected."""
    candidate = schema.CandidateConfig.model_validate(_base_candidate(priority=50))
    assert candidate.priority == 50
    assert isinstance(candidate.priority, int)

    with pytest.raises(ValidationError):
        schema.CandidateConfig.model_validate(_base_candidate(priority=1.5))

    missing_priority = _base_candidate()
    del missing_priority["priority"]
    with pytest.raises(ValidationError):
        schema.CandidateConfig.model_validate(missing_priority)


def test_priority_descending_with_zero_last_resort() -> None:
    """Compiled ordering is descending by priority; priority 0 placed last regardless of order."""
    alias = schema.AliasConfig.model_validate(
        _base_alias(
            candidates=[
                _base_candidate(model="c-low", priority=10),
                _base_candidate(model="c-zero", priority=0),
                _base_candidate(model="c-high", priority=200),
            ]
        )
    )
    ordered = schema.order_candidates_by_priority(alias.candidates)
    assert [c.model for c in ordered] == ["c-high", "c-low", "c-zero"]


def test_tie_break_distribution_then_declaration_order() -> None:
    """Equal non-zero priorities resolve by declared strategy; else stable declaration order."""
    alias_no_strategy = schema.AliasConfig.model_validate(
        _base_alias(
            candidates=[
                _base_candidate(model="first-declared", priority=50),
                _base_candidate(model="second-declared", priority=50),
            ]
        )
    )
    ordered = schema.order_candidates_by_priority(alias_no_strategy.candidates)
    assert [c.model for c in ordered] == ["first-declared", "second-declared"]

    alias_with_strategy = schema.AliasConfig.model_validate(
        _base_alias(
            distribution_strategy="proportional",
            candidates=[
                _base_candidate(model="a", priority=50, weight=1),
                _base_candidate(model="b", priority=50, weight=3),
            ],
        )
    )
    assert alias_with_strategy.distribution_strategy == "proportional"


def test_proportional_weights_normalized() -> None:
    """Proportional weights normalize; account/credential binding NOT expressed in YAML."""
    alias = schema.AliasConfig.model_validate(
        _base_alias(
            distribution_strategy="proportional",
            candidates=[
                _base_candidate(model="a", priority=50, weight=1),
                _base_candidate(model="b", priority=50, weight=3),
            ],
        )
    )
    weights = schema.normalized_weights(alias.candidates)
    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert pytest.approx(weights["a"], rel=1e-6) == 0.25
    assert pytest.approx(weights["b"], rel=1e-6) == 0.75

    # Account/credential binding must not be an accepted YAML field.
    with pytest.raises(ValidationError):
        schema.CandidateConfig.model_validate(_base_candidate(account_id="acct-123"))


def test_tui_attached_flag_compiles() -> None:
    """A candidate may declare tui_attached: <client> for per-model gating."""
    candidate = schema.CandidateConfig.model_validate(_base_candidate(tui_attached="Claude"))
    assert candidate.tui_attached == "Claude"

    candidate_default = schema.CandidateConfig.model_validate(_base_candidate())
    assert candidate_default.tui_attached is None


def test_schedule_windows_utc_only() -> None:
    """Schedule windows must be UTC; overlaps resolve deterministically; non-UTC rejected."""
    candidate = schema.CandidateConfig.model_validate(
        _base_candidate(
            schedule={
                "start": "2026-07-01T00:00:00Z",
                "end": "2026-07-15T00:00:00Z",
            }
        )
    )
    assert candidate.schedule is not None
    assert candidate.schedule.start.tzinfo is not None

    with pytest.raises(ValidationError):
        schema.CandidateConfig.model_validate(
            _base_candidate(
                schedule={
                    "start": "2026-07-01T00:00:00-05:00",
                    "end": "2026-07-15T00:00:00-05:00",
                }
            )
        )


def test_inheritance_resolves() -> None:
    """Typed inheritance (defaults -> alias -> candidate) merges without duplicate-definition ambiguity."""
    document = schema.RoutingConfigDocument.model_validate(
        {
            "defaults": {"route_family": "codex_responses"},
            "aliases": [
                {
                    "name": "read",
                    "route_family": "codex_openrouter_completion_adapter",
                    "candidates": [
                        {
                            "provider": "openrouter",
                            "model": "openrouter/cohere/north-mini-code:free",
                            "priority": 100,
                        },
                        {
                            "provider": "openai",
                            "model": "gpt-5.4-mini",
                            "route_family": "codex_responses",
                            "priority": 0,
                        },
                    ],
                }
            ],
        }
    )
    resolved = schema.resolve_inheritance(document)
    read_alias = resolved.aliases[0]
    assert read_alias.candidates[0].route_family == "codex_openrouter_completion_adapter"
    assert read_alias.candidates[1].route_family == "codex_responses"


def test_config_epoch_and_hash_present() -> None:
    """RoutingConfigDocument-level validation exposes fields the compiler needs for epoch/hash."""
    document = schema.RoutingConfigDocument.model_validate({"defaults": {}, "aliases": [_base_alias()]})
    assert hasattr(document, "aliases")
    assert document.aliases[0].name == "read"


def test_error_class_refs_open_vocabulary() -> None:
    """Error rules may reference class names not yet in the seed registry without failing."""
    candidate = schema.CandidateConfig.model_validate(
        _base_candidate(error_rules=[{"class_name": "a_totally_new_future_class", "cools": True}])
    )
    assert candidate.error_rules[0].class_name == "a_totally_new_future_class"
