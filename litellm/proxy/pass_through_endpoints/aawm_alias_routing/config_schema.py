"""Typed pydantic-v2 schema for the AAWM alias-routing YAML config (Wave 3, D1-583).

Owned by ``aawm_alias_routing`` package. This module defines the raw,
validated document shape (``RoutingConfigDocument`` -> ``AliasConfig`` ->
``CandidateConfig``), typed defaults -> alias -> candidate inheritance, and
pure ordering/weighting helpers. ``config_compiler.py`` consumes this module
to produce the immutable ``RoutingSnapshot`` defined in ``config_snapshot.py``.

Validation intentionally treats ``provider`` and ``route_family`` as
*references* into a registered set of known code behaviors (mirrored from
``policy.py``'s provider constants and candidate-table ``route_family``
values) -- never as arbitrary strings that could be evaluated or dynamically
imported. Error-class references (``ErrorRuleConfig.class_name``) are an
OPEN vocabulary by design and are never checked against a closed registry.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import policy

# Registered provider identities. Mirrors the provider constants declared in
# policy.py -- referenced by value only, never eval'd.
REGISTERED_PROVIDERS: frozenset[str] = frozenset(
    {
        policy.CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        policy.CODEX_AUTO_AGENT_GOOGLE_PROVIDER,
        policy.CODEX_AUTO_AGENT_ANTIGRAVITY_PROVIDER,
        policy.CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
        policy.CODEX_AUTO_AGENT_XAI_PROVIDER,
        policy.CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
        policy.CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
        policy.OPENCODE_ZEN_PROVIDER,
        policy.ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER,
    }
)

# Registered route-family (dispatch adapter) identities. Mirrors the
# ``route_family`` string values already used across policy.py's candidate
# tables (both codex and anthropic auto-agent lanes).
REGISTERED_ROUTE_FAMILIES: frozenset[str] = frozenset(
    {
        "codex_responses",
        "codex_openrouter_completion_adapter",
        "codex_grok_native_responses_adapter",
        "codex_xai_oauth_responses_adapter",
        "codex_kimi_chat_completions_adapter",
        "codex_alibaba_token_plan_chat_completions_adapter",
        "codex_opencode_zen_adapter",
        "anthropic_messages",
        "anthropic_openai_responses_adapter",
        "anthropic_openrouter_completion_adapter",
        "anthropic_kimi_chat_completions_adapter",
        "anthropic_alibaba_token_plan_chat_completions_adapter",
        "anthropic_grok_native_responses_adapter",
        "anthropic_xai_oauth_responses_adapter",
        "anthropic_opencode_zen_responses_adapter",
        "anthropic_opencode_zen_completion_adapter",
    }
)

DistributionStrategy = Literal["proportional", "round_robin"]


def _require_registered_provider(value: str) -> str:
    if value not in REGISTERED_PROVIDERS:
        raise ValueError(f"provider {value!r} is not a registered code behavior")
    return value


def _require_registered_route_family(value: Optional[str]) -> Optional[str]:
    if value is not None and value not in REGISTERED_ROUTE_FAMILIES:
        raise ValueError(f"route_family {value!r} is not a registered code behavior")
    return value


class ScheduleWindowConfig(BaseModel):
    """A UTC-only schedule window (e.g. a promo period for a candidate)."""

    model_config = ConfigDict(extra="forbid")

    start: datetime
    end: datetime

    @field_validator("start", "end")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("schedule window times must be UTC (offset zero)")
        return value

    @model_validator(mode="after")
    def _require_end_after_start(self) -> "ScheduleWindowConfig":
        if self.end < self.start:
            raise ValueError(f"schedule window end ({self.end!r}) must not precede start ({self.start!r})")
        return self


class ErrorRuleConfig(BaseModel):
    """A candidate-scoped error-class reference. Open vocabulary by design."""

    model_config = ConfigDict(extra="forbid")

    class_name: str
    cools: bool = True


class CandidateConfig(BaseModel):
    """A single routing candidate within an alias."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    route_family: Optional[str] = None
    priority: int
    weight: float = 1.0
    tui_attached: Optional[str] = None
    schedule: Optional[ScheduleWindowConfig] = None
    error_rules: list[ErrorRuleConfig] = Field(default_factory=list)

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        return _require_registered_provider(value)

    @field_validator("route_family")
    @classmethod
    def _validate_route_family(cls, value: Optional[str]) -> Optional[str]:
        return _require_registered_route_family(value)

    @field_validator("weight")
    @classmethod
    def _require_non_negative_weight(cls, value: float) -> float:
        if value < 0:
            raise ValueError(f"candidate weight {value!r} must not be negative")
        return value


class AliasConfig(BaseModel):
    """A single alias (e.g. ``read``) with its ordered candidate set."""

    model_config = ConfigDict(extra="forbid")

    name: str
    candidates: list[CandidateConfig]
    route_family: Optional[str] = None
    distribution_strategy: Optional[DistributionStrategy] = None

    @field_validator("route_family")
    @classmethod
    def _validate_route_family(cls, value: Optional[str]) -> Optional[str]:
        return _require_registered_route_family(value)

    @field_validator("candidates")
    @classmethod
    def _require_non_empty_candidates(cls, value: list[CandidateConfig]) -> list[CandidateConfig]:
        if not value:
            raise ValueError("alias candidates must not be empty -- at least one candidate is required")
        return value

    @field_validator("candidates")
    @classmethod
    def _require_unique_models(cls, value: list[CandidateConfig]) -> list[CandidateConfig]:
        seen: set[str] = set()
        for candidate in value:
            if candidate.model in seen:
                raise ValueError(f"duplicate model {candidate.model!r} within a single alias's candidate list")
            seen.add(candidate.model)
        return value


class DefaultsConfig(BaseModel):
    """Document-level defaults inherited by aliases/candidates when unset."""

    model_config = ConfigDict(extra="forbid")

    route_family: Optional[str] = None

    @field_validator("route_family")
    @classmethod
    def _validate_route_family(cls, value: Optional[str]) -> Optional[str]:
        return _require_registered_route_family(value)


class RoutingConfigDocument(BaseModel):
    """The top-level validated YAML document."""

    model_config = ConfigDict(extra="forbid")

    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    aliases: list[AliasConfig]

    @field_validator("aliases")
    @classmethod
    def _require_unique_alias_names(cls, value: list[AliasConfig]) -> list[AliasConfig]:
        seen: set[str] = set()
        for alias in value:
            if alias.name in seen:
                raise ValueError(f"duplicate alias name {alias.name!r} in routing config document")
            seen.add(alias.name)
        return value


def order_candidates_by_priority(
    candidates: Sequence[CandidateConfig],
) -> list[CandidateConfig]:
    """Order candidates descending by priority; ``priority: 0`` always last.

    Ties among non-zero priorities preserve declared (input) order -- the
    distribution strategy (proportional/round_robin) governs *selection*
    among ties, not ordering; Python's stable sort already preserves
    declaration order for equal keys.
    """
    non_zero = [c for c in candidates if c.priority != 0]
    zero = [c for c in candidates if c.priority == 0]
    non_zero_sorted = sorted(non_zero, key=lambda c: c.priority, reverse=True)
    return non_zero_sorted + zero


def normalized_weights(candidates: Sequence[CandidateConfig]) -> dict[str, float]:
    """Normalize ``weight`` across the given candidates so they sum to 1.0."""
    total = sum(candidate.weight for candidate in candidates) or 1.0
    return {candidate.model: candidate.weight / total for candidate in candidates}


def resolve_inheritance(document: RoutingConfigDocument) -> RoutingConfigDocument:
    """Resolve typed inheritance: defaults -> alias -> candidate.

    Currently resolves ``route_family``: a candidate's own value wins if
    set; otherwise the alias's value; otherwise the document defaults'
    value. Values that make it through this chain were already validated
    against ``REGISTERED_ROUTE_FAMILIES`` at whichever level set them, so
    the copies below do not need to re-validate.
    """
    resolved_aliases: list[AliasConfig] = []
    for alias in document.aliases:
        alias_route_family = alias.route_family if alias.route_family is not None else document.defaults.route_family
        resolved_candidates: list[CandidateConfig] = []
        for candidate in alias.candidates:
            effective_route_family = (
                candidate.route_family if candidate.route_family is not None else alias_route_family
            )
            resolved_candidates.append(candidate.model_copy(update={"route_family": effective_route_family}))
        resolved_aliases.append(alias.model_copy(update={"candidates": resolved_candidates}))
    return document.model_copy(update={"aliases": resolved_aliases})
