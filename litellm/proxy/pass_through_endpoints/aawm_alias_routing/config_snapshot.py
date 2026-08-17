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
from types import MappingProxyType
from typing import Mapping, Optional


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
class AliasReference:
    """Reference to another alias as a weighted branch (CFG-009)."""

    alias_name: str
    priority: int
    weight: float
    tui_attached: Optional[str] = None
    tui_excluded: Optional[str] = None


@dataclass(frozen=True, slots=True)
class DispatchRule:
    """Compiled TUI-family dispatch rule (CFG-007)."""

    tui_family: str
    target_alias: str


@dataclass(frozen=True, slots=True)
class DispatchSnapshot:
    """Compiled TUI-dispatch configuration (CFG-007)."""

    by_tui: tuple[DispatchRule, ...]
    default: Optional[str]
    blocked_tui_families: tuple[str, ...] = ()


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
    anthropic_route_family: Optional[str] = None
    # CFG-006: optional authoritative candidate-level reasoning effort
    # (canonical none|minimal|low|medium|high|xhigh|max). When set it
    # replaces caller/TUI reasoning at shaping time; when None the caller's
    # intent is preserved.
    reasoning_effort: Optional[str] = None
    # CFG-008: optional per-model TUI exclusion (product name, version-
    # insensitive). When set, the candidate is gated out only for requests
    # from the matching identified client product; complementary to
    # ``tui_attached`` for mutually exclusive TUI tails.
    tui_excluded: Optional[str] = None


@dataclass(frozen=True, slots=True)
class RoutingAlias:
    """A single compiled alias with its ordered, weighted candidate tuple or dispatch."""

    name: str
    distribution_strategy: Optional[str]
    candidates: tuple[RoutingCandidate | AliasReference, ...]
    # CFG-007: optional TUI-dispatch rules
    dispatch: Optional[DispatchSnapshot] = None


@dataclass(frozen=True, slots=True)
class RoutingSnapshot:
    """Immutable compiled routing configuration.

    ``config_epoch`` is a monotonically increasing integer bumped on every
    successful compile; it is **telemetry ordering only** and MUST NOT be
    used to invalidate runtime routing state.  ``config_hash`` is a
    deterministic SHA-256 *semantic digest* of the fully validated,
    inheritance-resolved compiled representation. Snapshot-resolved cooldown
    keys instead use a stable tag scoped to owning alias, provider, model, and
    resolved route semantics; the durable state-key namespace itself is
    unchanged. Affinity payloads intentionally persist ``config_hash`` as
    compatibility metadata for continuation validation.
    ``config_version`` is a human-facing identity string (first 12 hex chars
    of ``config_hash``).
    ``source_hash`` (optional) is the SHA-256 of the raw source YAML,
    retained for diagnostics only -- it MUST NOT control routing state.
    ``config_epoch`` and ``source_hash`` remain process-local snapshot fields;
    snapshot identity is also surfaced by the refresh endpoint response.
    """

    aliases: Mapping[str, RoutingAlias]
    config_epoch: int
    config_hash: str
    config_version: str
    source_hash: Optional[str] = None

    def __post_init__(self) -> None:
        # ``aliases`` is typed as ``Mapping`` for callers, but callers may
        # still pass a plain ``dict`` at construction time (as
        # ``config_compiler.py`` does). Wrap it in a read-only view here so
        # ``snapshot.aliases["x"] = ...`` raises regardless of what concrete
        # mapping type the caller constructed the snapshot with. The frozen
        # dataclass blocks attribute *reassignment*; this closes the
        # remaining inner-mapping-mutability gap.
        if not isinstance(self.aliases, MappingProxyType):
            object.__setattr__(self, "aliases", MappingProxyType(dict(self.aliases)))


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
