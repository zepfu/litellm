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
import json
import itertools
import threading

import yaml

from . import config_schema as schema
from .config_snapshot import (
    ErrorRule,
    RoutingAlias,
    RoutingCandidate,
    RoutingSnapshot,
    ScheduleWindow,
)


class ConfigCompileError(Exception):
    """Raised when the routing config YAML cannot be parsed/compiled."""


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


def _compile_candidate(candidate: schema.CandidateConfig, weight: float) -> RoutingCandidate:
    schedule = (
        ScheduleWindow(start=candidate.schedule.start, end=candidate.schedule.end)
        if candidate.schedule is not None
        else None
    )
    error_rules = tuple(ErrorRule(class_name=rule.class_name, cools=rule.cools) for rule in candidate.error_rules)
    anthropic_rf = schema.resolve_anthropic_route_family(
        candidate.route_family,
        candidate.anthropic_route_family,
    )
    # Ambiguous route families (e.g. codex_opencode_zen_adapter) may compile
    # with anthropic_route_family=None; the Anthropic ingress dispatch path
    # fails closed when shaping such candidates.  Non-ambiguous, unmapped
    # families with no explicit override are a compile-time error.
    if anthropic_rf is None and (
        candidate.route_family is None
        or candidate.route_family not in schema.AMBIGUOUS_CODEX_ROUTE_FAMILIES
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
    return RoutingCandidate(
        provider=candidate.provider,
        model=candidate.model,
        route_family=candidate.route_family,
        priority=candidate.priority,
        weight=weight,
        tui_attached=candidate.tui_attached,
        schedule=schedule,
        error_rules=error_rules,
        anthropic_route_family=anthropic_rf,
        reasoning_effort=candidate.reasoning_effort,
    )


def _compile_alias(alias: schema.AliasConfig) -> RoutingAlias:
    ordered = schema.order_candidates_by_priority(alias.candidates)
    if alias.distribution_strategy == "proportional":
        weights_by_model = schema.normalized_weights(ordered)
    else:
        weights_by_model = {candidate.model: candidate.weight for candidate in ordered}
    compiled_candidates = tuple(
        _compile_candidate(candidate, weights_by_model[candidate.model]) for candidate in ordered
    )
    return RoutingAlias(
        name=alias.name,
        distribution_strategy=alias.distribution_strategy,
        candidates=compiled_candidates,
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
        for candidate in alias.candidates:
            canonical_candidates.append(
                {
                    "error_rules": [
                        {"class_name": r.class_name, "cools": r.cools}
                        for r in candidate.error_rules
                    ],
                    "model": candidate.model,
                    "priority": candidate.priority,
                    "provider": candidate.provider,
                    "anthropic_route_family": candidate.anthropic_route_family,
                    "route_family": candidate.route_family,
                    "reasoning_effort": candidate.reasoning_effort,
                    "schedule": (
                        {
                            "end": candidate.schedule.end.isoformat(),
                            "start": candidate.schedule.start.isoformat(),
                        }
                        if candidate.schedule is not None
                        else None
                    ),
                    "tui_attached": candidate.tui_attached,
                    "weight": candidate.weight,
                }
            )
        canonical_aliases.append(
            {
                "candidates": canonical_candidates,
                "distribution_strategy": alias.distribution_strategy,
                "name": alias.name,
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

    aliases = {alias.name: _compile_alias(alias) for alias in resolved.aliases}
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
