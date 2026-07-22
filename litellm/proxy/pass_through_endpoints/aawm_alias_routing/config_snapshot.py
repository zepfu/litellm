"""Immutable routing snapshot + process-local atomic-swap holder (Wave 3, D1-583).

``config_compiler.py`` produces a ``RoutingSnapshot`` from validated YAML;
this module owns the frozen snapshot dataclasses themselves plus a
process-local holder with an atomic swap primitive, used later by the
Wave 5 refresh endpoint (and by Wave 4's selector integration to read the
currently active snapshot).

No I/O here -- compilation and file/network access live in
``config_compiler.py`` and the refresh endpoint respectively.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass(frozen=True, slots=True)
class ScheduleWindow:
    """Compiled UTC schedule window."""

    start: datetime
    end: datetime


@dataclass(frozen=True, slots=True)
class ErrorRule:
    """Compiled candidate-scoped error-class reference (open vocabulary)."""

    class_name: str
    cools: bool


@dataclass(frozen=True, slots=True)
class RoutingCandidate:
    """A single compiled routing candidate."""

    provider: str
    model: str
    route_family: Optional[str]
    priority: int
    weight: float
    tui_attached: Optional[str]
    schedule: Optional[ScheduleWindow]
    error_rules: tuple[ErrorRule, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class RoutingAlias:
    """A single compiled alias with its ordered, weighted candidate tuple."""

    name: str
    distribution_strategy: Optional[str]
    candidates: tuple[RoutingCandidate, ...]


@dataclass(frozen=True, slots=True)
class RoutingSnapshot:
    """Immutable compiled routing configuration.

    ``config_epoch`` is a monotonically increasing integer bumped on every
    successful compile/activation (used by Wave 4 to invalidate stale
    candidate-scoped cooldown/affinity keys). ``config_hash`` is a content
    hash of the source YAML; ``config_version`` is a human-facing identity
    string. Neither is persisted -- both are in-memory-only snapshot
    identity fields surfaced by the (Wave 5) refresh endpoint response.
    """

    aliases: dict[str, RoutingAlias]
    config_epoch: int
    config_hash: str
    config_version: str


class RoutingSnapshotHolder:
    """Process-local holder for the active ``RoutingSnapshot``.

    ``swap`` is atomic with respect to concurrent readers: a lock guards the
    single reference reassignment, and ``get`` never observes a partially
    constructed snapshot because ``RoutingSnapshot`` instances are always
    fully built before being handed to ``swap``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: Optional[RoutingSnapshot] = None

    def get(self) -> Optional[RoutingSnapshot]:
        return self._active

    def swap(self, snapshot: RoutingSnapshot) -> Optional[RoutingSnapshot]:
        """Atomically activate ``snapshot``, returning the previous one (if any)."""
        with self._lock:
            previous = self._active
            self._active = snapshot
            return previous


# Module-level singleton holder for the process's active routing snapshot.
active_routing_snapshot_holder = RoutingSnapshotHolder()


def get_active_snapshot() -> Optional[RoutingSnapshot]:
    """Return the process-local active ``RoutingSnapshot``, if any.

    Thin convenience wrapper over ``active_routing_snapshot_holder.get()`` --
    surfaced at module scope so callers (e.g. the Wave 5 refresh endpoint and
    its tests) don't need to reach through the holder singleton directly.
    """
    return active_routing_snapshot_holder.get()
