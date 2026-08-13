"""Typed pydantic-v2 schema for the AAWM alias-routing YAML config (Wave 3, D1-583).

Owned by ``aawm_alias_routing`` package. This module defines the raw,
validated document shape (``RoutingConfigDocument`` -> ``AliasConfig`` ->
``CandidateConfig``), typed defaults -> alias -> candidate inheritance, and
pure ordering/weighting helpers. ``config_compiler.py`` consumes this module
to produce the immutable ``RoutingSnapshot`` defined in ``config_snapshot.py``.

Validation intentionally treats ``provider`` and ``route_family`` as
*references* into registered provider and adapter behaviors -- never as
arbitrary strings that could be evaluated or dynamically imported. Error-class
references (``ErrorRuleConfig.class_name``) are an OPEN vocabulary by design
and are never checked against a closed registry.
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
        policy.CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
        policy.CODEX_AUTO_AGENT_XAI_PROVIDER,
        policy.CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
        policy.CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
        policy.CODEX_AUTO_AGENT_COHERE_PROVIDER,
        policy.OPENCODE_ZEN_PROVIDER,
        policy.ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER,
    }
)

# Registered route-family (dispatch adapter) identities.
REGISTERED_ROUTE_FAMILIES: frozenset[str] = frozenset(
    {
        "codex_responses",
        "codex_openrouter_completion_adapter",
        "codex_grok_native_responses_adapter",
        "codex_xai_oauth_responses_adapter",
        "codex_kimi_chat_completions_adapter",
        "codex_alibaba_token_plan_chat_completions_adapter",
        "codex_cohere_chat_completions_adapter",
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


# Closed ingress-specific route-family projection: maps each Codex/OpenAI
# Responses ingress route_family to its Anthropic Messages ingress equivalent.
# Codex-only candidates are intentionally absent from this mapping.
CODEX_TO_ANTHROPIC_ROUTE_FAMILY: dict[str, str] = {
    "codex_responses": "anthropic_openai_responses_adapter",
    "codex_openrouter_completion_adapter": "anthropic_openrouter_completion_adapter",
    "codex_grok_native_responses_adapter": "anthropic_grok_native_responses_adapter",
    "codex_xai_oauth_responses_adapter": "anthropic_xai_oauth_responses_adapter",
    "codex_kimi_chat_completions_adapter": "anthropic_kimi_chat_completions_adapter",
    "codex_alibaba_token_plan_chat_completions_adapter": "anthropic_alibaba_token_plan_chat_completions_adapter",
}

CODEX_ONLY_ROUTE_FAMILIES: frozenset[str] = frozenset(
    {
        "codex_cohere_chat_completions_adapter",
    }
)

# Route families that are ambiguous across ingress (one codex family maps to
# multiple possible anthropic families depending on the specific model/candidate).
# These REQUIRE an explicit ``anthropic_route_family`` per candidate.
AMBIGUOUS_CODEX_ROUTE_FAMILIES: frozenset[str] = frozenset(
    {
        "codex_opencode_zen_adapter",
    }
)


def resolve_anthropic_route_family(
    codex_route_family: Optional[str],
    explicit_override: Optional[str],
) -> Optional[str]:
    """Resolve the effective Anthropic-ingress route family for a candidate.

    Priority: explicit override > closed mapping > None (fail closed at compile).
    """
    if explicit_override is not None:
        return explicit_override
    if codex_route_family is not None:
        return CODEX_TO_ANTHROPIC_ROUTE_FAMILY.get(codex_route_family)
    return None


DistributionStrategy = Literal[
    "proportional",
    "round_robin",
    "highest_quota_available",
    "lowest_quota_available",
]


# Canonical reasoning-effort vocabulary accepted at the candidate YAML level
# (CFG-006). Matches the tier order used by the shared reasoning-effort
# normalization seams; values are stored verbatim and translated per provider
# route at dispatch time.
REGISTERED_REASONING_EFFORTS: frozenset[str] = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)

# TUI family normalization vocabulary (CFG-007 dispatch).
REGISTERED_TUI_FAMILIES: frozenset[str] = frozenset(
    {"codex", "claude", "grok", "qwen", "kimi", "unknown"}
)


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

class AliasReferenceCandidateConfig(BaseModel):
    """Reference to another alias as a weighted branch (CFG-009).

    Alias references are weighted at branch level and exhausted as a unit
    before the parent marks the branch unavailable and reselects.
    """

    model_config = ConfigDict(extra="forbid")

    alias_reference: str
    priority: int = 1
    weight: float = 1.0
    tui_attached: Optional[str] = None
    tui_excluded: Optional[str] = None

    @field_validator("weight")
    @classmethod
    def _require_non_negative_weight(cls, value: float) -> float:
        if value < 0:
            raise ValueError(f"alias_reference weight {value!r} must not be negative")
        return value


class DispatchRuleConfig(BaseModel):
    """TUI-family dispatch rule for alias-to-alias routing (CFG-007)."""

    model_config = ConfigDict(extra="forbid")

    tui_family: str
    target_alias: str

    @field_validator("tui_family")
    @classmethod
    def _require_registered_tui_family(cls, value: str) -> str:
        if value not in REGISTERED_TUI_FAMILIES:
            raise ValueError(f"tui_family {value!r} is not a registered TUI family")
        return value


class DispatchConfig(BaseModel):
    """TUI-origin dispatch configuration for logical aliases (CFG-007).

    by_tui maps TUI family names to target aliases. default is used
    when no TUI rule matches or when the origin is unknown/missing.
    """

    model_config = ConfigDict(extra="forbid")

    by_tui: list[DispatchRuleConfig] = Field(default_factory=list)
    default: Optional[str] = None
    blocked_tui_families: list[str] = Field(default_factory=list)

    @field_validator("blocked_tui_families")
    @classmethod
    def _require_registered_blocked_tui_families(
        cls, value: list[str]
    ) -> list[str]:
        invalid = [family for family in value if family not in REGISTERED_TUI_FAMILIES]
        if invalid:
            raise ValueError(
                f"blocked_tui_families contains unregistered TUI families: {invalid!r}"
            )
        if len(set(value)) != len(value):
            raise ValueError("blocked_tui_families must not contain duplicates")
        return value


class CandidateConfig(BaseModel):
    """A single routing candidate within an alias."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    route_family: Optional[str] = None
    anthropic_route_family: Optional[str] = None
    priority: int
    weight: float = 1.0
    tui_attached: Optional[str] = None
    # CFG-008: optional per-model TUI exclusion. When set, the candidate is
    # ineligible only for requests whose identified client product name
    # equals this value; missing/unknown origins remain eligible. Matching
    # mirrors ``tui_attached`` (product name only, version-insensitive).
    tui_excluded: Optional[str] = None
    schedule: Optional[ScheduleWindowConfig] = None
    error_rules: list[ErrorRuleConfig] = Field(default_factory=list)
    reasoning_effort: Optional[str] = None

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        return _require_registered_provider(value)

    @field_validator("route_family")
    @classmethod
    def _validate_route_family(cls, value: Optional[str]) -> Optional[str]:
        return _require_registered_route_family(value)

    @field_validator("anthropic_route_family")
    @classmethod
    def _validate_anthropic_route_family(cls, value: Optional[str]) -> Optional[str]:
        return _require_registered_route_family(value)

    @field_validator("weight")
    @classmethod
    def _require_non_negative_weight(cls, value: float) -> float:
        if value < 0:
            raise ValueError(f"candidate weight {value!r} must not be negative")
        return value

    @field_validator("reasoning_effort")
    @classmethod
    def _require_canonical_reasoning_effort(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in REGISTERED_REASONING_EFFORTS:
            raise ValueError(
                f"candidate reasoning_effort {value!r} is not a canonical effort value; "
                f"expected one of none|minimal|low|medium|high|xhigh|max"
            )
        return value


class AliasConfig(BaseModel):
    """A single alias with its candidate set or dispatch rules."""

    model_config = ConfigDict(extra="forbid")

    name: str
    candidates: list[CandidateConfig | AliasReferenceCandidateConfig] = Field(
        default_factory=list
    )
    route_family: Optional[str] = None
    distribution_strategy: Optional[DistributionStrategy] = None
    # CFG-007: optional TUI-dispatch rules for logical aliases like sota.
    dispatch: Optional[DispatchConfig] = None

    @field_validator("route_family")
    @classmethod
    def _validate_route_family(cls, value: Optional[str]) -> Optional[str]:
        return _require_registered_route_family(value)

    @field_validator("candidates")
    @classmethod
    def _require_unique_entries(cls, value: list) -> list:
        seen: set[str] = set()
        for entry in value:
            if isinstance(entry, CandidateConfig):
                if entry.model in seen:
                    raise ValueError(
                        f"duplicate model {entry.model!r} within a single alias"
                    )
                seen.add(entry.model)
            elif isinstance(entry, AliasReferenceCandidateConfig):
                if entry.alias_reference in seen:
                    raise ValueError(
                        f"duplicate alias_reference {entry.alias_reference!r}"
                    )
                seen.add(entry.alias_reference)
        return value

    @model_validator(mode="after")
    def _require_candidates_or_dispatch(self) -> "AliasConfig":
        if not self.candidates and self.dispatch is None:
            raise ValueError(
                f"alias {self.name!r} must have either candidates or dispatch"
            )
        if self.candidates and self.dispatch is not None:
            raise ValueError(
                f"alias {self.name!r} cannot have both candidates and dispatch"
            )
        return self

    @model_validator(mode="after")
    def _validate_distribution_contract(self) -> "AliasConfig":
        if self.distribution_strategy in {
            "highest_quota_available",
            "lowest_quota_available",
        }:
            weighted = [
                entry
                for entry in self.candidates
                if float(getattr(entry, "weight", 1.0)) != 1.0
            ]
            if weighted:
                raise ValueError(
                    "quota availability strategies require default weight 1.0 "
                    "for every candidate or alias reference"
                )
        return self


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
            normalized_name = alias.name.casefold()
            if normalized_name in seen:
                raise ValueError(f"duplicate alias name {alias.name!r} in routing config document")
            seen.add(normalized_name)
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


def order_alias_entries_by_priority(
    entries: Sequence[CandidateConfig | AliasReferenceCandidateConfig],
) -> list[CandidateConfig | AliasReferenceCandidateConfig]:
    """Order mixed concrete/reference entries by priority with zero last."""
    non_zero = [entry for entry in entries if entry.priority != 0]
    zero = [entry for entry in entries if entry.priority == 0]
    return sorted(non_zero, key=lambda entry: entry.priority, reverse=True) + zero


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
        alias_route_family = (
            alias.route_family
            if alias.route_family is not None
            else document.defaults.route_family
        )
        resolved_candidates: list[CandidateConfig | AliasReferenceCandidateConfig] = []
        for candidate in alias.candidates:
            if isinstance(candidate, AliasReferenceCandidateConfig):
                resolved_candidates.append(candidate)
                continue
            effective_route_family = (
                candidate.route_family if candidate.route_family is not None else alias_route_family
            )
            resolved_candidates.append(
                candidate.model_copy(update={"route_family": effective_route_family})
            )
        resolved_aliases.append(
            alias.model_copy(update={"candidates": resolved_candidates})
        )
    return document.model_copy(update={"aliases": resolved_aliases})


def detect_alias_reference_cycles(document: RoutingConfigDocument) -> list[str]:
    """Detect cycles across alias references and dispatch targets.

    Returns list of cycle path descriptions. Raises ValueError for
    missing-target references or dispatch targets.
    """
    alias_map = {alias.name: alias for alias in document.aliases}

    def _walk(name: str, path: list[str]) -> Optional[str]:
        if name in path:
            return " -> ".join(path + [name])
        alias = alias_map.get(name)
        if alias is None:
            raise ValueError(
                f"alias_reference {name!r} not found in config document"
            )
        path = path + [name]
        targets = [
            candidate.alias_reference
            for candidate in alias.candidates
            if isinstance(candidate, AliasReferenceCandidateConfig)
        ]
        if alias.dispatch is not None:
            targets.extend(rule.target_alias for rule in alias.dispatch.by_tui)
            if alias.dispatch.default is not None:
                targets.append(alias.dispatch.default)
        for target in targets:
            result = _walk(target, path)
            if result is not None:
                return result
        return None

    cycles: list[str] = []
    seen_roots: set[str] = set()
    for alias_name in alias_map:
        cycle = _walk(alias_name, [])
        if cycle is not None:
            # Deduplicate: only report once per cycle set
            cycle_members = set(cycle.split(" -> "))
            if not cycle_members & seen_roots:
                cycles.append(cycle)
                seen_roots.update(cycle_members)
    return cycles
