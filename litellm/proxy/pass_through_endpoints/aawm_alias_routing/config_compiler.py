"""Compile validated AAWM alias-routing YAML into an immutable snapshot (Wave 3, D1-583).

``compile_yaml`` is the single public entry point: parse YAML -> validate
against ``config_schema.RoutingConfigDocument`` -> resolve typed inheritance
-> order candidates by descending numeric priority (``priority: 0`` reserved
last-resort, placed last) -> normalize proportional weights -> produce a
frozen ``config_snapshot.RoutingSnapshot`` carrying ``config_epoch``,
``config_hash``, and ``config_version``.

Malformed YAML (parse failure) raises ``ConfigCompileError``. Schema/
reference validation failures (unknown keys, unregistered provider/
route_family, non-typed priority, etc.) raise ``pydantic.ValidationError``
directly from ``config_schema`` -- both signal a rejected compile with no
partial/silent activation.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import threading
from collections.abc import Mapping
from datetime import timedelta
from typing import Optional

import yaml

from . import config_schema as schema
from .config_snapshot import (
    AliasReference,
    DispatchRule,
    DispatchSnapshot,
    ErrorRule,
    RoutingAlias,
    RoutingCandidate,
    RoutingSnapshot,
    ScheduleWindow,
)


class ConfigCompileError(Exception):
    """Raised when the routing config YAML cannot be parsed/compiled."""


PROVIDER_ALIAS_PREFIX = "provider-"

# Closed route-family vocabulary per registered provider. Provider-pinned
# aliases may only use families from this map; a newly registered provider
# stays uncovered until both an alias and an allowed-family entry exist.
_PROVIDER_ALLOWED_ROUTE_FAMILIES: dict[str, frozenset[str]] = {
    "openai": frozenset(
        {
            "codex_responses",
            "anthropic_openai_responses_adapter",
        }
    ),
    "anthropic": frozenset({"anthropic_messages"}),
    "openrouter": frozenset(
        {
            "codex_openrouter_completion_adapter",
            "anthropic_openrouter_completion_adapter",
        }
    ),
    "xai": frozenset(
        {
            "codex_xai_oauth_responses_adapter",
            "anthropic_xai_oauth_responses_adapter",
            "codex_grok_native_responses_adapter",
            "anthropic_grok_native_responses_adapter",
        }
    ),
    "kimi_code": frozenset(
        {
            "codex_kimi_chat_completions_adapter",
            "anthropic_kimi_chat_completions_adapter",
        }
    ),
    "alibaba_token_plan": frozenset(
        {
            "codex_alibaba_token_plan_chat_completions_adapter",
            "anthropic_alibaba_token_plan_chat_completions_adapter",
        }
    ),
    "zai_coding_plan": frozenset(
        {
            "codex_zai_coding_plan_chat_completions_adapter",
        }
    ),
    "cohere": frozenset({"codex_cohere_chat_completions_adapter"}),
    "nous": frozenset({"codex_nous_chat_completions_adapter"}),
    "cursor_agent": frozenset(
        {
            "codex_cursor_agent_aiserver_adapter",
            "anthropic_cursor_agent_aiserver_adapter",
        }
    ),
    "nvidia": frozenset({"codex_nvidia_completion_adapter"}),
    "opencode_zen": frozenset(
        {
            "codex_opencode_zen_adapter",
            "anthropic_opencode_zen_responses_adapter",
            "anthropic_opencode_zen_completion_adapter",
        }
    ),
    "opencode_go": frozenset({"codex_opencode_go_adapter"}),
}


def provider_alias_name(provider_id: str) -> str:
    """Return the canonical ``provider-<id>`` alias spelling."""

    return f"{PROVIDER_ALIAS_PREFIX}{provider_id}"


def iter_provider_alias_names(aliases: Mapping[str, object]) -> tuple[str, ...]:
    """Return sorted ``provider-<id>`` alias names present in *aliases*."""

    return tuple(sorted(name for name in aliases if name.startswith(PROVIDER_ALIAS_PREFIX)))


def uncovered_registered_providers(aliases: Mapping[str, object]) -> tuple[str, ...]:
    """Return registered providers without a configured ``provider-<id>`` alias.

    Direct provider registration does not require a provider-pinned alias.
    """

    present = {
        name[len(PROVIDER_ALIAS_PREFIX) :]
        for name in iter_provider_alias_names(aliases)
    }
    return tuple(sorted(schema.REGISTERED_PROVIDERS - present))


def _assert_provider_alias_coverage(aliases: dict[str, RoutingAlias]) -> None:
    """Reject a compile that introduces an incomplete provider-alias inventory.

    Documents without any ``provider-*`` alias skip this check so focused
    unit YAML stays valid. Once any provider alias is present, each configured
    provider alias must be a closed same-provider alias.
    """

    provider_names = iter_provider_alias_names(aliases)
    if not provider_names:
        return

    errors: list[str] = []
    seen_providers: dict[str, str] = {}
    for name in provider_names:
        provider_id = name[len(PROVIDER_ALIAS_PREFIX) :]
        if provider_id not in schema.REGISTERED_PROVIDERS:
            errors.append(
                f"alias {name!r} does not name a registered provider"
            )
            continue
        if provider_id in seen_providers:
            errors.append(
                f"duplicate provider alias {name!r} (already {seen_providers[provider_id]!r})"
            )
            continue
        seen_providers[provider_id] = name
        alias = aliases[name]
        if alias.dispatch is not None:
            errors.append(
                f"{name} uses TUI dispatch; provider aliases must be closed candidate sets"
            )
        concrete = 0
        for entry in alias.candidates:
            if isinstance(entry, AliasReference):
                errors.append(
                    f"{name} uses alias_reference {entry.alias_name!r}; "
                    "provider aliases must not escape via alias_reference"
                )
                continue
            concrete += 1
            if entry.provider != provider_id:
                errors.append(
                    f"{name} candidate {entry.model!r} has provider "
                    f"{entry.provider!r}, expected {provider_id!r}"
                )
            allowed = _PROVIDER_ALLOWED_ROUTE_FAMILIES.get(provider_id)
            if allowed is None:
                errors.append(
                    f"registered provider {provider_id!r} has no allowed "
                    "route-family vocabulary for provider aliases"
                )
                continue
            for route_family in (entry.route_family, entry.anthropic_route_family):
                if route_family is not None and route_family not in allowed:
                    errors.append(
                        f"{name} candidate {entry.model!r} route family "
                        f"{route_family!r} is not an allowed {provider_id} family"
                    )
        if concrete == 0:
            errors.append(f"{name} has no concrete same-provider candidates")

    missing_vocab = sorted(
        provider_id
        for provider_id in seen_providers
        if provider_id not in _PROVIDER_ALLOWED_ROUTE_FAMILIES
    )
    if missing_vocab:
        errors.append(
            "configured providers lack provider-alias route-family vocabulary: "
            + ", ".join(missing_vocab)
        )
    if errors:
        raise ConfigCompileError(
            "provider-alias inventory incomplete: " + "; ".join(errors)
        )


_epoch_lock = threading.Lock()
_epoch_counter = itertools.count(1)


def _next_epoch() -> int:
    """Return a process-wide monotonically increasing compile epoch."""
    with _epoch_lock:
        return next(_epoch_counter)


# Provider / route-family credential compatibility.  A candidate whose
# route_family implies a specific credential domain MUST NOT be paired
# with a provider from an incompatible domain.  This prevents e.g.
# routing an Anthropic/Claude model through Codex/OpenAI credentials
# (a TOS violation) or an OpenAI model through Anthropic Messages
# credentials.
_CODEX_CREDENTIAL_ROUTE_FAMILIES: frozenset[str] = frozenset(
    {
        "codex_responses",
    }
)
_ANTHROPIC_CREDENTIAL_ROUTE_FAMILIES: frozenset[str] = frozenset(
    {
        "anthropic_messages",
    }
)
_ANTHROPIC_NATIVE_PROVIDERS: frozenset[str] = frozenset(
    {
        "anthropic",
    }
)
_OPENAI_NATIVE_PROVIDERS: frozenset[str] = frozenset(
    {
        "openai",
    }
)
_COHERE_CREDENTIAL_ROUTE_FAMILIES: frozenset[str] = frozenset(
    {
        "codex_cohere_chat_completions_adapter",
    }
)
_COHERE_NATIVE_PROVIDERS: frozenset[str] = frozenset(
    {
        "cohere",
    }
)
_NOUS_CREDENTIAL_ROUTE_FAMILIES: frozenset[str] = frozenset(
    {
        "codex_nous_chat_completions_adapter",
    }
)
_NOUS_NATIVE_PROVIDERS: frozenset[str] = frozenset(
    {
        "nous",
    }
)
_ZAI_CODING_PLAN_CREDENTIAL_ROUTE_FAMILIES: frozenset[str] = frozenset(
    {
        "codex_zai_coding_plan_chat_completions_adapter",
    }
)
_ZAI_CODING_PLAN_NATIVE_PROVIDERS: frozenset[str] = frozenset(
    {
        "zai_coding_plan",
    }
)
_CURSOR_AGENT_CREDENTIAL_ROUTE_FAMILIES: frozenset[str] = frozenset(
    {
        "codex_cursor_agent_aiserver_adapter",
        "anthropic_cursor_agent_aiserver_adapter",
    }
)
_CURSOR_AGENT_NATIVE_PROVIDERS: frozenset[str] = frozenset(
    {
        "cursor_agent",
    }
)
_NVIDIA_CREDENTIAL_ROUTE_FAMILIES: frozenset[str] = frozenset(
    {
        "codex_nvidia_completion_adapter",
    }
)
_NVIDIA_NATIVE_PROVIDERS: frozenset[str] = frozenset(
    {
        "nvidia",
    }
)


def _validate_cohere_credential_domain(
    *,
    provider: str,
    model: str,
    route_family: Optional[str],
    anthropic_route_family: Optional[str],
) -> None:
    route_families = tuple(
        value
        for value in (route_family, anthropic_route_family)
        if value is not None
    )
    uses_cohere_credentials = any(
        value in _COHERE_CREDENTIAL_ROUTE_FAMILIES for value in route_families
    )
    is_cohere_provider = provider in _COHERE_NATIVE_PROVIDERS

    if uses_cohere_credentials and not is_cohere_provider:
        raise ConfigCompileError(
            f"candidate model {model!r}: provider {provider!r} is incompatible "
            "with Cohere-credential route family"
        )
    if is_cohere_provider and any(
        value not in _COHERE_CREDENTIAL_ROUTE_FAMILIES for value in route_families
    ):
        raise ConfigCompileError(
            f"candidate model {model!r}: provider {provider!r} requires "
            "Cohere-native route families"
        )


def _validate_nous_credential_domain(
    *,
    provider: str,
    model: str,
    route_family: Optional[str],
    anthropic_route_family: Optional[str],
) -> None:
    route_families = tuple(
        value
        for value in (route_family, anthropic_route_family)
        if value is not None
    )
    uses_nous_credentials = any(
        value in _NOUS_CREDENTIAL_ROUTE_FAMILIES for value in route_families
    )
    is_nous_provider = provider in _NOUS_NATIVE_PROVIDERS

    if uses_nous_credentials and not is_nous_provider:
        raise ConfigCompileError(
            f"candidate model {model!r}: provider {provider!r} is incompatible "
            "with Nous-credential route family"
        )
    if is_nous_provider and any(
        value not in _NOUS_CREDENTIAL_ROUTE_FAMILIES for value in route_families
    ):
        raise ConfigCompileError(
            f"candidate model {model!r}: provider {provider!r} requires "
            "Nous-native route families"
        )


def _validate_zai_coding_plan_credential_domain(
    *,
    provider: str,
    model: str,
    route_family: Optional[str],
    anthropic_route_family: Optional[str],
) -> None:
    route_families = tuple(
        value
        for value in (route_family, anthropic_route_family)
        if value is not None
    )
    uses_coding_plan_credentials = any(
        value in _ZAI_CODING_PLAN_CREDENTIAL_ROUTE_FAMILIES for value in route_families
    )
    is_coding_plan_provider = provider in _ZAI_CODING_PLAN_NATIVE_PROVIDERS

    if uses_coding_plan_credentials and not is_coding_plan_provider:
        raise ConfigCompileError(
            f"candidate model {model!r}: provider {provider!r} is incompatible "
            "with Z.AI Coding Plan credential route family"
        )
    if is_coding_plan_provider and any(
        value not in _ZAI_CODING_PLAN_CREDENTIAL_ROUTE_FAMILIES for value in route_families
    ):
        raise ConfigCompileError(
            f"candidate model {model!r}: provider {provider!r} requires "
            "Z.AI Coding Plan-native route families"
        )


def _validate_cursor_agent_credential_domain(
    *,
    provider: str,
    model: str,
    route_family: Optional[str],
    anthropic_route_family: Optional[str],
) -> None:
    route_families = tuple(
        value
        for value in (route_family, anthropic_route_family)
        if value is not None
    )
    uses_cursor_agent_credentials = any(
        value in _CURSOR_AGENT_CREDENTIAL_ROUTE_FAMILIES for value in route_families
    )
    is_cursor_agent_provider = provider in _CURSOR_AGENT_NATIVE_PROVIDERS

    if uses_cursor_agent_credentials and not is_cursor_agent_provider:
        raise ConfigCompileError(
            f"candidate model {model!r}: provider {provider!r} is incompatible "
            "with Cursor Agent credential route family"
        )
    if is_cursor_agent_provider and any(
        value not in _CURSOR_AGENT_CREDENTIAL_ROUTE_FAMILIES for value in route_families
    ):
        raise ConfigCompileError(
            f"candidate model {model!r}: provider {provider!r} requires "
            "Cursor Agent route families"
        )


def _validate_nvidia_credential_domain(
    *,
    provider: str,
    model: str,
    route_family: Optional[str],
    anthropic_route_family: Optional[str],
) -> None:
    route_families = tuple(
        value
        for value in (route_family, anthropic_route_family)
        if value is not None
    )
    uses_nvidia_credentials = any(
        value in _NVIDIA_CREDENTIAL_ROUTE_FAMILIES for value in route_families
    )
    is_nvidia_provider = provider in _NVIDIA_NATIVE_PROVIDERS

    if uses_nvidia_credentials and not is_nvidia_provider:
        raise ConfigCompileError(
            f"candidate model {model!r}: provider {provider!r} is incompatible "
            "with NVIDIA-credential route family"
        )
    if is_nvidia_provider and any(
        value not in _NVIDIA_CREDENTIAL_ROUTE_FAMILIES for value in route_families
    ):
        raise ConfigCompileError(
            f"candidate model {model!r}: provider {provider!r} requires "
            "NVIDIA-native route families"
        )


def _format_fixed_utc_offset(offset: timedelta) -> str:
    total_minutes = int(offset.total_seconds() // 60)
    sign = "-" if total_minutes < 0 else "+"
    hours, minutes = divmod(abs(total_minutes), 60)
    return f"{sign}{hours:02d}:{minutes:02d}"


def _compile_schedule(
    schedule: Optional[schema.ScheduleWindowConfig],
) -> Optional[ScheduleWindow]:
    if schedule is None:
        return None
    if schedule.kind == "absolute":
        return ScheduleWindow(
            kind="absolute",
            start=schedule.start,
            end=schedule.end,
        )
    return ScheduleWindow(
        kind="daily",
        start_time=schedule.start_time,
        end_time=schedule.end_time,
        utc_offset=schedule.utc_offset,
        timezone=schedule.timezone,
    )


def _canonical_schedule_repr(
    schedule: Optional[ScheduleWindow],
) -> Optional[dict[str, object]]:
    if schedule is None:
        return None
    if schedule.kind == "absolute":
        assert schedule.start is not None
        assert schedule.end is not None
        return {
            "end": schedule.end.isoformat(),
            "kind": "absolute",
            "start": schedule.start.isoformat(),
        }
    assert schedule.start_time is not None
    assert schedule.end_time is not None
    if schedule.timezone is not None:
        return {
            "end_time": schedule.end_time.isoformat(),
            "kind": "daily",
            "start_time": schedule.start_time.isoformat(),
            "timezone": schedule.timezone,
        }
    assert schedule.utc_offset is not None
    return {
        "end_time": schedule.end_time.isoformat(),
        "kind": "daily",
        "start_time": schedule.start_time.isoformat(),
        "utc_offset": _format_fixed_utc_offset(schedule.utc_offset),
    }


def _compile_candidate(candidate: schema.CandidateConfig, weight: float) -> RoutingCandidate:
    schedule = _compile_schedule(candidate.schedule)
    error_rules = tuple(ErrorRule(class_name=rule.class_name, cools=rule.cools) for rule in candidate.error_rules)
    anthropic_rf = schema.resolve_anthropic_route_family(
        candidate.route_family,
        candidate.anthropic_route_family,
    )
    if (
        candidate.route_family in schema.CODEX_ONLY_ROUTE_FAMILIES
        and candidate.anthropic_route_family is not None
    ):
        raise ConfigCompileError(
            f"candidate model {candidate.model!r}: Codex-only route family "
            f"{candidate.route_family!r} cannot set anthropic_route_family"
        )
    # Ambiguous route families (e.g. codex_opencode_zen_adapter) may compile
    # with anthropic_route_family=None; the Anthropic ingress dispatch path
    # fails closed when shaping such candidates.  Non-ambiguous, unmapped
    # families with no explicit override are a compile-time error, except for
    # route families explicitly marked Codex-only.
    if anthropic_rf is None and (
        candidate.route_family is None
        or (
            candidate.route_family not in schema.AMBIGUOUS_CODEX_ROUTE_FAMILIES
            and candidate.route_family not in schema.CODEX_ONLY_ROUTE_FAMILIES
        )
    ):
        raise ConfigCompileError(
            f"candidate model {candidate.model!r} has no resolvable Anthropic-ingress "
            f"route family: codex route_family {candidate.route_family!r} is not in the "
            f"closed projection and no explicit anthropic_route_family override was provided"
        )
    # Credential-domain compatibility: reject provider/route pairings that
    # would require cross-provider credentials at dispatch time.
    if candidate.route_family in _CODEX_CREDENTIAL_ROUTE_FAMILIES and (
        candidate.provider in _ANTHROPIC_NATIVE_PROVIDERS
    ):
        raise ConfigCompileError(
            f"candidate model {candidate.model!r}: provider {candidate.provider!r} "
            f"is incompatible with codex-credential route_family {candidate.route_family!r}"
        )
    if anthropic_rf in _ANTHROPIC_CREDENTIAL_ROUTE_FAMILIES and (
        candidate.provider in _OPENAI_NATIVE_PROVIDERS
    ):
        raise ConfigCompileError(
            f"candidate model {candidate.model!r}: provider {candidate.provider!r} "
            f"is incompatible with anthropic-credential route_family {anthropic_rf!r}"
        )
    _validate_cohere_credential_domain(
        provider=candidate.provider,
        model=candidate.model,
        route_family=candidate.route_family,
        anthropic_route_family=anthropic_rf,
    )
    _validate_nous_credential_domain(
        provider=candidate.provider,
        model=candidate.model,
        route_family=candidate.route_family,
        anthropic_route_family=anthropic_rf,
    )
    _validate_zai_coding_plan_credential_domain(
        provider=candidate.provider,
        model=candidate.model,
        route_family=candidate.route_family,
        anthropic_route_family=anthropic_rf,
    )
    _validate_cursor_agent_credential_domain(
        provider=candidate.provider,
        model=candidate.model,
        route_family=candidate.route_family,
        anthropic_route_family=anthropic_rf,
    )
    _validate_nvidia_credential_domain(
        provider=candidate.provider,
        model=candidate.model,
        route_family=candidate.route_family,
        anthropic_route_family=anthropic_rf,
    )
    return RoutingCandidate(
        provider=candidate.provider,
        model=candidate.model,
        route_family=candidate.route_family,
        priority=candidate.priority,
        weight=weight,
        tui_attached=candidate.tui_attached,
        tui_excluded=candidate.tui_excluded,
        schedule=schedule,
        error_rules=error_rules,
        anthropic_route_family=anthropic_rf,
        reasoning_effort=candidate.reasoning_effort,
    )



def _compile_alias_reference(
    alias_ref: schema.AliasReferenceCandidateConfig,
    *,
    available_aliases: set[str],
) -> AliasReference:
    """Compile an alias reference, validating the target exists."""
    if alias_ref.alias_reference not in available_aliases:
        raise ConfigCompileError(
            f"alias_reference {alias_ref.alias_reference!r} not found in config document"
        )
    return AliasReference(
        alias_name=alias_ref.alias_reference,
        priority=alias_ref.priority,
        weight=alias_ref.weight,
        tui_attached=alias_ref.tui_attached,
        tui_excluded=alias_ref.tui_excluded,
        schedule=_compile_schedule(alias_ref.schedule),
    )


def _compile_dispatch(
    dispatch: Optional[schema.DispatchConfig],
    *,
    available_aliases: set[str],
) -> Optional[DispatchSnapshot]:
    """Compile TUI-dispatch rules (CFG-007)."""
    if dispatch is None:
        return None
    compiled_rules = []
    for rule in dispatch.by_tui:
        if rule.target_alias not in available_aliases:
            raise ConfigCompileError(
                f"dispatch rule for tui_family {rule.tui_family!r}: "
                f"target_alias {rule.target_alias!r} not found"
            )
        compiled_rules.append(
            DispatchRule(
                tui_family=rule.tui_family,
                target_alias=rule.target_alias,
            )
        )
    if dispatch.default is not None and dispatch.default not in available_aliases:
        raise ConfigCompileError(
            f"dispatch default {dispatch.default!r} not found in config document"
        )
    return DispatchSnapshot(
        by_tui=tuple(compiled_rules),
        default=dispatch.default,
        blocked_tui_families=tuple(dispatch.blocked_tui_families),
    )


def _compile_alias(
    alias: schema.AliasConfig, *, available_aliases: set[str]
) -> RoutingAlias:
    if not alias.candidates and alias.dispatch is not None:
        dispatch_snapshot = _compile_dispatch(
            alias.dispatch, available_aliases=available_aliases
        )
        return RoutingAlias(
            name=alias.name,
            distribution_strategy=alias.distribution_strategy,
            candidates=(),
            dispatch=dispatch_snapshot,
            multi_agent_version=alias.multi_agent_version,
        )

    ordered_entries = schema.order_alias_entries_by_priority(alias.candidates)
    concrete_candidates = [
        entry
        for entry in ordered_entries
        if isinstance(entry, schema.CandidateConfig)
    ]
    ordered = schema.order_candidates_by_priority(concrete_candidates)
    if alias.distribution_strategy == "proportional":
        weights_by_model = schema.normalized_weights(ordered)
    else:
        weights_by_model = {
            candidate.model: candidate.weight for candidate in ordered
        }

    compiled_entries: list[RoutingCandidate | AliasReference] = []
    for entry in ordered_entries:
        if isinstance(entry, schema.CandidateConfig):
            compiled_entries.append(
                _compile_candidate(entry, weights_by_model[entry.model])
            )
        else:
            compiled_entries.append(
                _compile_alias_reference(
                    entry, available_aliases=available_aliases
                )
            )

    dispatch_snapshot = _compile_dispatch(
        alias.dispatch, available_aliases=available_aliases
    )

    return RoutingAlias(
        name=alias.name,
        distribution_strategy=alias.distribution_strategy,
        candidates=tuple(compiled_entries),
        dispatch=dispatch_snapshot,
        multi_agent_version=alias.multi_agent_version,
    )


def _canonical_snapshot_repr(aliases: dict[str, RoutingAlias]) -> str:
    """Build a deterministic canonical JSON string from compiled aliases.

    Aliases are sorted by name (mapping-key order is not semantic).
    Candidate list ordering IS preserved (it encodes priority / round-robin
    fallback semantics).  All mapping keys within each object are sorted via
    json.dumps(sort_keys=True) so that YAML mapping-key order, comments,
    and whitespace never affect the digest.
    """
    canonical_aliases: list[dict[str, object]] = []
    for name in sorted(aliases):
        alias = aliases[name]
        canonical_candidates: list[dict[str, object]] = []
        for entry in alias.candidates:
            if isinstance(entry, AliasReference):
                canonical_candidates.append(
                    {
                        "type": "alias_reference",
                        "alias_name": entry.alias_name,
                        "priority": entry.priority,
                        "schedule": _canonical_schedule_repr(entry.schedule),
                        "tui_attached": entry.tui_attached,
                        "tui_excluded": entry.tui_excluded,
                        "weight": entry.weight,
                    }
                )
            else:  # RoutingCandidate
                canonical_candidates.append(
                    {
                        "type": "candidate",
                        "error_rules": [
                            {"class_name": r.class_name, "cools": r.cools}
                            for r in entry.error_rules
                        ],
                        "model": entry.model,
                        "priority": entry.priority,
                        "provider": entry.provider,
                        "anthropic_route_family": entry.anthropic_route_family,
                        "route_family": entry.route_family,
                        "reasoning_effort": entry.reasoning_effort,
                        "schedule": _canonical_schedule_repr(entry.schedule),
                        "tui_attached": entry.tui_attached,
                        "tui_excluded": entry.tui_excluded,
                        "weight": entry.weight,
                    }
                )

        dispatch_repr = None
        if alias.dispatch is not None:
            dispatch_repr = {
                "by_tui": [
                    {"tui_family": rule.tui_family, "target_alias": rule.target_alias}
                    for rule in alias.dispatch.by_tui
                ],
                "default": alias.dispatch.default,
                "blocked_tui_families": list(
                    alias.dispatch.blocked_tui_families
                ),
            }

        canonical_aliases.append(
            {
                "candidates": canonical_candidates,
                "distribution_strategy": alias.distribution_strategy,
                "name": alias.name,
                "dispatch": dispatch_repr,
                "multi_agent_version": alias.multi_agent_version,
            }
        )
    return json.dumps({"aliases": canonical_aliases}, sort_keys=True, separators=(",", ":"))


def compile_yaml(raw_yaml: str) -> RoutingSnapshot:
    """Validate and compile ``raw_yaml`` into an immutable ``RoutingSnapshot``."""
    try:
        raw_data = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as exc:
        raise ConfigCompileError(f"invalid YAML: {exc}") from exc

    if not isinstance(raw_data, dict):
        raise ConfigCompileError("routing config document must be a mapping")

    document = schema.RoutingConfigDocument.model_validate(raw_data)
    resolved = schema.resolve_inheritance(document)

    try:
        cycles = schema.detect_alias_reference_cycles(resolved)
    except ValueError as exc:
        raise ConfigCompileError(str(exc)) from exc
    if cycles:
        raise ConfigCompileError(f"alias delegation cycle detected: {'; '.join(cycles)}")

    available_aliases = {alias.name for alias in resolved.aliases}
    aliases = {
        alias.name: _compile_alias(alias, available_aliases=available_aliases)
        for alias in resolved.aliases
    }
    _assert_provider_alias_coverage(aliases)
    source_hash = hashlib.sha256(raw_yaml.encode("utf-8")).hexdigest()
    semantic_repr = _canonical_snapshot_repr(aliases)
    config_hash = hashlib.sha256(semantic_repr.encode("utf-8")).hexdigest()
    config_version = config_hash[:12]

    return RoutingSnapshot(
        aliases=aliases,
        config_epoch=_next_epoch(),
        config_hash=config_hash,
        config_version=config_version,
        source_hash=source_hash,
    )
