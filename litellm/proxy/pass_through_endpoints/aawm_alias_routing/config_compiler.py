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


def _compile_candidate(candidate: schema.CandidateConfig, weight: float) -> RoutingCandidate:
    schedule = (
        ScheduleWindow(start=candidate.schedule.start, end=candidate.schedule.end)
        if candidate.schedule is not None
        else None
    )
    error_rules = tuple(ErrorRule(class_name=rule.class_name, cools=rule.cools) for rule in candidate.error_rules)
    return RoutingCandidate(
        provider=candidate.provider,
        model=candidate.model,
        route_family=candidate.route_family,
        priority=candidate.priority,
        weight=weight,
        tui_attached=candidate.tui_attached,
        schedule=schedule,
        error_rules=error_rules,
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
    config_hash = hashlib.sha256(raw_yaml.encode("utf-8")).hexdigest()
    config_version = config_hash[:12]

    return RoutingSnapshot(
        aliases=aliases,
        config_epoch=_next_epoch(),
        config_hash=config_hash,
        config_version=config_version,
    )
