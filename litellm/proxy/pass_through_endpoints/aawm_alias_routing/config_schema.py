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

import re
from collections import defaultdict
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta
from heapq import heappop, heappush
from typing import Literal, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
        policy.CODEX_AUTO_AGENT_ZAI_CODING_PLAN_PROVIDER,
        policy.CODEX_AUTO_AGENT_COHERE_PROVIDER,
        policy.CODEX_AUTO_AGENT_NOUS_PROVIDER,
        policy.CODEX_AUTO_AGENT_CURSOR_AGENT_PROVIDER,
        policy.CODEX_AUTO_AGENT_NVIDIA_PROVIDER,
        policy.CODEX_AUTO_AGENT_MUSE_CODE_PROVIDER,
        policy.OPENCODE_ZEN_PROVIDER,
        policy.OPENCODE_GO_PROVIDER,
        policy.ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER,
    }
)

# Registered route-family (dispatch adapter) identities.
REGISTERED_ROUTE_FAMILIES: frozenset[str] = frozenset(
    {
        "codex_responses",
        "codex_openrouter_completion_adapter",
        policy.CODEX_AUTO_AGENT_OPENROUTER_RESPONSES_ROUTE_FAMILY,
        "codex_grok_native_responses_adapter",
        "codex_xai_oauth_responses_adapter",
        "codex_kimi_chat_completions_adapter",
        "codex_alibaba_token_plan_chat_completions_adapter",
        "codex_zai_coding_plan_chat_completions_adapter",
        "codex_cohere_chat_completions_adapter",
        "codex_nous_chat_completions_adapter",
        "codex_cursor_agent_aiserver_adapter",
        "codex_nvidia_completion_adapter",
        "codex_opencode_zen_adapter",
        "codex_opencode_go_adapter",
        "muse_code",
        "anthropic_messages",
        "anthropic_openai_responses_adapter",
        "anthropic_openrouter_completion_adapter",
        "anthropic_kimi_chat_completions_adapter",
        "anthropic_alibaba_token_plan_chat_completions_adapter",
        "anthropic_grok_native_responses_adapter",
        "anthropic_xai_oauth_responses_adapter",
        "anthropic_opencode_zen_responses_adapter",
        "anthropic_opencode_zen_completion_adapter",
        "anthropic_cursor_agent_aiserver_adapter",
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
    "codex_cursor_agent_aiserver_adapter": "anthropic_cursor_agent_aiserver_adapter",
}

CODEX_ONLY_ROUTE_FAMILIES: frozenset[str] = frozenset(
    {
        "codex_cohere_chat_completions_adapter",
        "codex_nous_chat_completions_adapter",
        "codex_zai_coding_plan_chat_completions_adapter",
        "codex_opencode_go_adapter",
        "codex_nvidia_completion_adapter",
        policy.CODEX_AUTO_AGENT_OPENROUTER_RESPONSES_ROUTE_FAMILY,
        "muse_code",
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
REGISTERED_REASONING_EFFORTS: frozenset[str] = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})

# TUI family normalization vocabulary (CFG-007 dispatch).
# `ohmypi` is a first-class origin (Oh My Pi / ompla / omp). It is not a
# `sota.yaml` by_tui target; logical `sota` still uses `default` for it.
REGISTERED_TUI_FAMILIES: frozenset[str] = frozenset({"codex", "claude", "grok", "qwen", "kimi", "ohmypi", "unknown"})

# Bounds for the supported alias graph. References and dispatch both count
# as an alias hop. Expansion counts retain repeated occurrences because a
# reference is a weighted/fallback branch, not a semantic deduplication point.
MAX_ALIAS_GRAPH_DEPTH = 64
MAX_ALIAS_EXPANDED_CANDIDATES = 4096
MAX_ALIAS_EXPANSION_WORK = 32768


def _require_registered_provider(value: str) -> str:
    if value not in REGISTERED_PROVIDERS:
        raise ValueError(f"provider {value!r} is not a registered code behavior")
    return value


def _require_registered_route_family(value: Optional[str]) -> Optional[str]:
    if value is not None and value not in REGISTERED_ROUTE_FAMILIES:
        raise ValueError(f"route_family {value!r} is not a registered code behavior")
    return value


def _parse_local_clock_time(value: object) -> time:
    if isinstance(value, time):
        if value.tzinfo is not None:
            raise ValueError("daily schedule clock times must not include a timezone")
        return value.replace(microsecond=0)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("daily schedule clock times must be HH:MM or HH:MM:SS")
    raw = value.strip()
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            parsed = datetime.strptime(raw, fmt).time()
        except ValueError:
            continue
        return parsed.replace(microsecond=0)
    raise ValueError("daily schedule clock times must be HH:MM or HH:MM:SS")


def _parse_fixed_utc_offset(value: object) -> timedelta:
    if isinstance(value, timedelta):
        offset = value
    elif isinstance(value, str) and value.strip():
        raw = value.strip()
        if raw.upper() in {"UTC", "Z"}:
            offset = timedelta(0)
        else:
            match = None
            for pattern in (
                r"^UTC(?P<sign>[+-])(?P<hours>\d{1,2})(?::?(?P<minutes>\d{2}))?$",
                r"^(?P<sign>[+-])(?P<hours>\d{1,2})(?::(?P<minutes>\d{2}))?$",
            ):
                match = re.fullmatch(pattern, raw, flags=re.IGNORECASE)
                if match is not None:
                    break
            if match is None:
                raise ValueError("daily schedule utc_offset must be a fixed offset such as +08:00")
            hours = int(match.group("hours"))
            minutes = int(match.group("minutes") or 0)
            if hours > 18 or minutes > 59:
                raise ValueError("daily schedule utc_offset is out of range")
            sign = -1 if match.group("sign") == "-" else 1
            offset = timedelta(hours=hours, minutes=minutes) * sign
    else:
        raise ValueError("daily schedule utc_offset must be a fixed offset such as +08:00")
    if offset.total_seconds() % 60:
        raise ValueError("daily schedule utc_offset must be a whole-minute offset")
    return offset


def _parse_timezone(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("daily schedule timezone must be an IANA timezone such as " "America/Los_Angeles")
    raw = value.strip()
    try:
        ZoneInfo(raw)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError(f"daily schedule timezone {raw!r} is not a valid IANA timezone") from exc
    return raw


_WEEKDAY_NAMES = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _parse_schedule_date(value: object) -> date:
    if type(value) is date:
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError("daily schedule dates must use YYYY/MM/DD")
    try:
        return datetime.strptime(value.strip(), "%Y/%m/%d").date()
    except ValueError as exc:
        raise ValueError("daily schedule dates must use YYYY/MM/DD") from exc


def _parse_weekdays(value: object) -> tuple[int, ...]:
    if value is None:
        return None
    if isinstance(value, (str, bytes, int)) or not isinstance(value, Sequence):
        raise ValueError("daily schedule weekdays must be a list")
    weekdays: set[int] = set()
    for item in value:
        if isinstance(item, bool):
            raise ValueError(
                "daily schedule weekdays must be Monday=0 through Sunday=6 " "or an unabbreviated English weekday name"
            )
        if isinstance(item, int):
            if item not in range(7):
                raise ValueError("daily schedule weekdays must be Monday=0 through Sunday=6")
            weekdays.add(item)
            continue
        if not isinstance(item, str) or item.strip().lower() not in _WEEKDAY_NAMES:
            raise ValueError(
                "daily schedule weekdays must be Monday=0 through Sunday=6 " "or an unabbreviated English weekday name"
            )
        weekdays.add(_WEEKDAY_NAMES[item.strip().lower()])
    if not weekdays:
        raise ValueError("daily schedule weekdays must contain at least one day")
    return tuple(sorted(weekdays))


class ScheduleWindowConfig(BaseModel):
    """Absolute UTC, whole-day calendar, or daily local-time window."""

    model_config = ConfigDict(extra="forbid")

    start: Optional[datetime] = None
    end: Optional[datetime] = None
    start_time: Optional[time] = None
    end_time: Optional[time] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    utc_offset: Optional[timedelta] = None
    timezone: Optional[str] = None
    weekdays: Optional[tuple[int, ...]] = None

    @field_validator("start", "end")
    @classmethod
    def _require_utc(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("schedule window times must be UTC (offset zero)")
        return value

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def _parse_daily_clock(cls, value: object) -> object:
        if value is None:
            return None
        return _parse_local_clock_time(value)

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def _parse_daily_date(cls, value: object) -> object:
        if value is None:
            return None
        return _parse_schedule_date(value)

    @field_validator("weekdays", mode="before")
    @classmethod
    def _parse_daily_weekdays(cls, value: object) -> object:
        return _parse_weekdays(value)

    @field_validator("utc_offset", mode="before")
    @classmethod
    def _parse_daily_offset(cls, value: object) -> object:
        if value is None:
            return None
        return _parse_fixed_utc_offset(value)

    @field_validator("timezone", mode="before")
    @classmethod
    def _parse_daily_timezone(cls, value: object) -> object:
        if value is None:
            return None
        return _parse_timezone(value)

    @model_validator(mode="after")
    def _require_exclusive_schedule_shape(self) -> "ScheduleWindowConfig":
        has_absolute = self.start is not None or self.end is not None
        has_daily = (
            self.start_time is not None
            or self.end_time is not None
            or self.start_date is not None
            or self.end_date is not None
            or self.utc_offset is not None
            or self.timezone is not None
            or self.weekdays is not None
        )
        if has_absolute and has_daily:
            raise ValueError(
                "schedule cannot mix absolute start/end with daily " "start_time/end_time/utc_offset/timezone"
            )
        if has_absolute:
            if self.start is None or self.end is None:
                raise ValueError("absolute schedule windows require both start and end")
            if self.end < self.start:
                raise ValueError(f"schedule window end ({self.end!r}) must not precede start ({self.start!r})")
            return self
        if (self.start_date is None) != (self.end_date is None):
            raise ValueError("daily schedule date ranges require start_date and end_date")
        if self.start_date is not None and self.end_date < self.start_date:
            raise ValueError(
                f"daily schedule end_date ({self.end_date!r}) must not precede " f"start_date ({self.start_date!r})"
            )
        if self.utc_offset is not None and self.timezone is not None:
            raise ValueError("daily schedule windows require either utc_offset or timezone, " "not both")
        has_clocks = self.start_time is not None or self.end_time is not None
        has_calendar_gate = self.start_date is not None or self.end_date is not None or self.weekdays is not None
        if has_clocks and (self.start_time is None or self.end_time is None):
            raise ValueError("daily schedule clock windows require start_time and end_time")
        if has_clocks and (self.utc_offset is None and self.timezone is None):
            raise ValueError(
                "daily schedule windows require start_time, end_time, and " "either utc_offset or timezone"
            )
        if not has_clocks and not has_calendar_gate:
            raise ValueError("daily schedule windows require clocks or whole-day calendar gates")
        return self

    @property
    def kind(self) -> Literal["absolute", "daily"]:
        if self.start is not None and self.end is not None:
            return "absolute"
        return "daily"


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
    schedule: Optional[ScheduleWindowConfig] = None

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
    def _require_registered_blocked_tui_families(cls, value: list[str]) -> list[str]:
        invalid = [family for family in value if family not in REGISTERED_TUI_FAMILIES]
        if invalid:
            raise ValueError(f"blocked_tui_families contains unregistered TUI families: {invalid!r}")
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
    candidates: list[CandidateConfig | AliasReferenceCandidateConfig] = Field(default_factory=list)
    route_family: Optional[str] = None
    distribution_strategy: Optional[DistributionStrategy] = None
    # CURSOR-014: optional stock Codex collaboration metadata for the alias
    # catalog. This is intentionally alias-scoped rather than candidate-scoped.
    multi_agent_version: Optional[Literal["v1", "v2"]] = None
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
                    raise ValueError(f"duplicate model {entry.model!r} within a single alias")
                seen.add(entry.model)
            elif isinstance(entry, AliasReferenceCandidateConfig):
                if entry.alias_reference in seen:
                    raise ValueError(f"duplicate alias_reference {entry.alias_reference!r}")
                seen.add(entry.alias_reference)
        return value

    @model_validator(mode="after")
    def _require_candidates_or_dispatch(self) -> "AliasConfig":
        if not self.candidates and self.dispatch is None:
            raise ValueError(f"alias {self.name!r} must have either candidates or dispatch")
        if self.candidates and self.dispatch is not None:
            raise ValueError(f"alias {self.name!r} cannot have both candidates and dispatch")
        return self

    @model_validator(mode="after")
    def _validate_distribution_contract(self) -> "AliasConfig":
        if self.distribution_strategy in {
            "highest_quota_available",
            "lowest_quota_available",
        }:
            weighted = [entry for entry in self.candidates if float(getattr(entry, "weight", 1.0)) != 1.0]
            if weighted:
                raise ValueError(
                    "quota availability strategies require default weight 1.0 " "for every candidate or alias reference"
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
        alias_route_family = alias.route_family if alias.route_family is not None else document.defaults.route_family
        resolved_candidates: list[CandidateConfig | AliasReferenceCandidateConfig] = []
        for candidate in alias.candidates:
            if isinstance(candidate, AliasReferenceCandidateConfig):
                resolved_candidates.append(candidate)
                continue
            effective_route_family = (
                candidate.route_family if candidate.route_family is not None else alias_route_family
            )
            resolved_candidates.append(candidate.model_copy(update={"route_family": effective_route_family}))
        resolved_aliases.append(alias.model_copy(update={"candidates": resolved_candidates}))
    return document.model_copy(update={"aliases": resolved_aliases})


def detect_alias_reference_cycles(document: RoutingConfigDocument) -> list[str]:
    """Detect cycles across alias references and dispatch targets.

    Returns list of cycle path descriptions. Raises ValueError for
    missing-target references or dispatch targets.
    """
    alias_map = {alias.name: alias for alias in document.aliases}
    edges: dict[str, list[str]] = {}
    for alias_name, alias in alias_map.items():
        targets = [
            candidate.alias_reference
            for candidate in alias.candidates
            if isinstance(candidate, AliasReferenceCandidateConfig)
        ]
        if alias.dispatch is not None:
            targets.extend(rule.target_alias for rule in alias.dispatch.by_tui)
            if alias.dispatch.default is not None:
                targets.append(alias.dispatch.default)
        edges[alias_name] = targets

    active: set[str] = set()
    completed: set[str] = set()

    for alias_name in alias_map:
        if alias_name in completed:
            continue

        # Each frame is [node, next-edge-index]. The explicit path is popped
        # when its frame completes, while ``completed`` records descendants
        # that have already been checked and need no second traversal.
        path: list[str] = []
        stack: list[list] = [[alias_name, 0]]
        active.add(alias_name)
        while stack:
            frame = stack[-1]
            node = frame[0]
            if node in active and node not in path:
                alias = alias_map.get(node)
                if alias is None:
                    raise ValueError(
                        f"alias_reference {node!r} not found in config document"
                    )
                path.append(node)

            if frame[1] >= len(edges[node]):
                active.discard(node)
                completed.add(node)
                path.pop()
                stack.pop()
                continue

            target = edges[node][frame[1]]
            frame[1] += 1
            if target in active:
                cycle_start = path.index(target)
                return [" -> ".join(path[cycle_start:] + [target])]
            if target in completed:
                continue
            if target not in edges:
                raise ValueError(
                    f"alias_reference {target!r} not found in config document"
                )
            active.add(target)
            stack.append([target, 0])

    return []


def _alias_graph_edges(
    document: RoutingConfigDocument,
) -> tuple[
    dict[str, "AliasConfig"],
    dict[str, list[str]],
    dict[str, int],
    dict[str, set[str]],
]:
    """Build direct-reference, candidate, and unique-target graph inputs."""
    alias_map = {alias.name: alias for alias in document.aliases}
    references: dict[str, list[str]] = {}
    direct_candidates: dict[str, int] = {}
    dependencies: dict[str, set[str]] = defaultdict(set)

    for alias_name, alias in alias_map.items():
        references[alias_name] = [
            candidate.alias_reference
            for candidate in alias.candidates
            if isinstance(candidate, AliasReferenceCandidateConfig)
        ]
        direct_candidates[alias_name] = sum(
            isinstance(candidate, CandidateConfig) for candidate in alias.candidates
        )
        dependencies[alias_name].update(references[alias_name])
        if alias.dispatch is not None:
            dependencies[alias_name].update(
                rule.target_alias for rule in alias.dispatch.by_tui
            )
            if alias.dispatch.default is not None:
                dependencies[alias_name].add(alias.dispatch.default)
    return alias_map, references, direct_candidates, dependencies


def _count_alias_graph_expansion(
    alias_map: dict[str, "AliasConfig"],
    references: dict[str, list[str]],
    direct_candidates: dict[str, int],
    dependencies: dict[str, set[str]],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Count each alias expansion in bounded dependency order.

    Shared DAG descendants are counted once, while repeated branch occurrences
    remain represented in the expansion total. Dispatch has one selected
    target at runtime, so its branch metrics use the largest target.
    """
    dependents: dict[str, set[str]] = defaultdict(set)
    remaining = {alias_name: len(targets) for alias_name, targets in dependencies.items()}
    for alias_name, targets in dependencies.items():
        for target in targets:
            dependents[target].add(alias_name)

    candidate_count: dict[str, int] = {}
    work_count: dict[str, int] = {}
    depth_count: dict[str, int] = {}
    output_cap = MAX_ALIAS_EXPANDED_CANDIDATES + 1
    work_cap = MAX_ALIAS_EXPANSION_WORK + 1
    depth_cap = MAX_ALIAS_GRAPH_DEPTH + 1
    pending = [alias_name for alias_name, count in remaining.items() if count == 0]
    ready = set(pending)

    # Heap by document order keeps rejection deterministic without recursion.
    while pending:
        alias_name = heappop(pending)
        ready.discard(alias_name)
        alias = alias_map[alias_name]
        is_dispatch = alias.dispatch is not None
        # Dependencies are unique for topological ordering, but reference
        # occurrences must remain distinct in the expanded output and work
        # budgets.
        targets = dependencies[alias_name] if is_dispatch else references[alias_name]

        output = direct_candidates[alias_name]
        work = 1
        depth = 1
        for target in targets:
            if is_dispatch:
                output = max(output, candidate_count[target])
                work = max(work, 1 + work_count[target])
                depth = max(depth, 1 + depth_count[target])
            else:
                output += candidate_count[target]
                work += work_count[target]
                depth = max(depth, 1 + depth_count[target])

        candidate_count[alias_name] = min(output, output_cap)
        work_count[alias_name] = min(work, work_cap)
        depth_count[alias_name] = min(depth, depth_cap)

        for parent in dependents[alias_name]:
            remaining[parent] -= 1
            if remaining[parent] == 0 and parent not in ready:
                heappush(pending, parent)
                ready.add(parent)

    return candidate_count, work_count, depth_count


def _first_alias_budget_overflow(
    alias_map: dict[str, "AliasConfig"],
    candidate_count: dict[str, int],
    work_count: dict[str, int],
    depth_count: dict[str, int],
) -> Optional[tuple[str, str, int, int]]:
    for alias_name in alias_map:
        if depth_count[alias_name] > MAX_ALIAS_GRAPH_DEPTH:
            return (
                alias_name,
                "depth",
                depth_count[alias_name],
                MAX_ALIAS_GRAPH_DEPTH,
            )
        if candidate_count[alias_name] > MAX_ALIAS_EXPANDED_CANDIDATES:
            return (
                alias_name,
                "expanded-candidate",
                candidate_count[alias_name],
                MAX_ALIAS_EXPANDED_CANDIDATES,
            )
        if work_count[alias_name] > MAX_ALIAS_EXPANSION_WORK:
            return (
                alias_name,
                "expansion-work",
                work_count[alias_name],
                MAX_ALIAS_EXPANSION_WORK,
            )
    return None


def validate_alias_graph_bounds(document: RoutingConfigDocument) -> None:
    """Reject alias graphs beyond the documented publication budget.

    Counting is bounded and does not materialize the expansion. Shared DAG
    descendants are counted once in dependency order, while their repeated
    occurrences remain represented in the expansion total.
    """
    alias_map, references, direct_candidates, dependencies = _alias_graph_edges(document)
    counts = _count_alias_graph_expansion(
        alias_map, references, direct_candidates, dependencies
    )
    overflow = _first_alias_budget_overflow(alias_map, *counts)
    if overflow is not None:
        alias_name, budget, actual, maximum = overflow
        raise ValueError(
            f"alias graph {alias_name!r} exceeds maximum {budget} budget: "
            f"{actual} > {maximum}"
        )
